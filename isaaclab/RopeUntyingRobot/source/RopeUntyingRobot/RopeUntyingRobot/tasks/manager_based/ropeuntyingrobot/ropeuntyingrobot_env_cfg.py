"""Environment configuration for the RopeReach-SO100 task.

Phase 1: Move the SO-100 end-effector to the rope centre-of-mass.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, DeformableObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from . import mdp

# ---------------------------------------------------------------------------
# Asset path resolution — the SO-100 config lives in the assets/ directory
# two levels above the extension source root.
# ---------------------------------------------------------------------------
_ASSET_DIR = (
    Path(__file__).resolve().parents[6] / "assets"
)

# Lazy import: so100_config sits in the assets directory alongside the URDF.
import sys as _sys

if str(_ASSET_DIR) not in _sys.path:
    _sys.path.insert(0, str(_ASSET_DIR))

from so100_config import create_so100_rl_articulation_cfg  # noqa: E402


def _build_so100_cfg():
    """Build the SO-100 ArticulationCfg with RL actuator split."""
    return create_so100_rl_articulation_cfg(asset_dir=_ASSET_DIR)


# ============================================================================
# Scene
# ============================================================================


@configclass
class RopeReachSceneCfg(InteractiveSceneCfg):
    """SO-100 arm + table + deformable rope scene."""

    # -- ground --
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.CuboidCfg(
            size=(4.0, 4.0, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.18, 0.18, 0.18),
                roughness=0.9,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.025)),
    )

    # -- table --
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.4, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.45, 0.35, 0.25),
                roughness=0.8,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.25, 0.0, -0.02)),
    )

    # -- deformable rope --
    rope = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Rope",
        spawn=sim_utils.MeshCylinderCfg(
            radius=0.00175,
            height=0.45,
            axis="X",
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                self_collision=True,
                solver_position_iteration_count=32,
                vertex_velocity_damping=0.05,
                simulation_hexahedral_resolution=10,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                density=700.0,
                youngs_modulus=2000000.0,
                poissons_ratio=0.40,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 1.0),
                roughness=0.7,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.25, 0.0, 0.05),
        ),
    )

    # -- SO-100 robot --
    robot = _build_so100_cfg().replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    # -- end-effector frame tracker --
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper",
                name="ee",
            ),
        ],
        debug_vis=False,
    )

    # -- lighting --
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.85, 0.85, 0.85),
        ),
    )


# ============================================================================
# MDP — Actions
# ============================================================================


@configclass
class ActionsCfg:
    """5-DOF joint position control on the arm. Gripper stays open."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ],
        scale=0.5,
        use_default_offset=True,
    )


# ============================================================================
# MDP — Observations (21-D)
# ============================================================================


@configclass
class ObservationsCfg:
    """Observation specification for the reaching task."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations fed to the policy network."""

        # -- joint state (5 + 5) --
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "shoulder_pan",
                        "shoulder_lift",
                        "elbow_flex",
                        "wrist_flex",
                        "wrist_roll",
                    ],
                ),
            },
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "shoulder_pan",
                        "shoulder_lift",
                        "elbow_flex",
                        "wrist_flex",
                        "wrist_roll",
                    ],
                ),
            },
        )

        # -- rope COM in world frame (3) --
        rope_com = ObsTerm(func=mdp.rope_com_pos)

        # -- EE position in world frame (3) --
        ee_pos = ObsTerm(func=mdp.ee_pos_w)

        # -- last action (5) --
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ============================================================================
# MDP — Events (resets)
# ============================================================================


@configclass
class EventCfg:
    """Reset events for robot joints and rope position."""

    reset_arm_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                ],
            ),
            "position_range": (0.8, 1.2),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_rope = EventTerm(
        func=mdp.reset_rope_randomized,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("rope"),
            "position_offset_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
            },
            "rotation_range": (0.0, 6.283185307),
            "per_node_noise_std": 0.01,
        },
    )


# ============================================================================
# MDP — Rewards
# ============================================================================


@configclass
class RewardsCfg:
    """Reward terms for the reaching task."""

    # Dense reaching reward
    reaching_rope = RewTerm(
        func=mdp.reaching_rope,
        weight=1.0,
        params={"sigma": 0.1},
    )

    # Binary bonus for being close
    close_to_rope = RewTerm(
        func=mdp.close_to_rope,
        weight=5.0,
        params={"threshold": 0.02},
    )

    # Regularisation: smooth actions
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

    # Regularisation: low joint velocities
    joint_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.001,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                ],
            ),
        },
    )


# ============================================================================
# MDP — Terminations
# ============================================================================


@configclass
class TerminationsCfg:
    """Termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    rope_fell = DoneTerm(
        func=mdp.rope_below_height,
        params={"min_height": -0.05},
    )


# ============================================================================
# Full environment config
# ============================================================================


@configclass
class RopeReachEnvCfg(ManagerBasedRLEnvCfg):
    """RopeReach-SO100-v0: move the gripper to the rope COM."""

    scene: RopeReachSceneCfg = RopeReachSceneCfg(
        num_envs=64,
        env_spacing=2.5,
        replicate_physics=False,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # Simulation at 120 Hz, control at 30 Hz (decimation=4)
        self.decimation = 4
        self.episode_length_s = 10.0
        self.viewer.eye = (1.0, 1.0, 0.8)
        self.viewer.lookat = (0.25, 0.0, 0.1)
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
