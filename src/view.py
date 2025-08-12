import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib as mpl

CURRENT_TIME = 16

def add_cube(cube_definition, ax, color="b", edgecolor="k", alpha=0.2):
    cube_definition_array = [np.array(list(item)) for item in cube_definition]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0],
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]],
    ]

    faces = Poly3DCollection(
        edges, linewidths=1, edgecolors=edgecolor, facecolors=color, alpha=alpha
    )

    ax.add_collection3d(faces)
    # Plot the points themselves to force the scaling of the axes
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0)

def shift_cuboid(x_shift, y_shift, cuboid):
    cuboid = np.copy(cuboid)
    cuboid[:, 0] += x_shift
    cuboid[:, 1] += y_shift

    return cuboid

def rotate_point_zaxis(p, angle):
    rot_matrix = np.array(
        [
            [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0],
            [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0],
            [0, 0, 1],
        ]
    )
    return np.matmul(p, rot_matrix)

def rotate_bbox_zaxis(bbox, angle):
    bbox = np.copy(bbox)
    _bbox = []
    angle = np.rad2deg(-angle)
    for point in bbox:
        _bbox.append(rotate_point_zaxis(point, angle))

    return np.array(_bbox)

car = np.array(
    [
        (-2.25, -1, 0),  # left bottom front
        (-2.25, 1, 0),  # left bottom back
        (2.25, -1, 0),  # right bottom front
        (-2.25, -1, 1.5),  # left top front -> height
    ]
)

def plot_scenario(e2ed_data):

    routing = ["driving straight", "turning left" , "turning right"]

    traj = e2ed_data["agent/pos"] # (36,2)
    intent = e2ed_data["agent/intent"][0] #(1,)

    current_pos = traj[CURRENT_TIME-1] # (2,)

    fig = plt.figure(figsize=(15, 15), dpi=80)

    front_view = np.concatenate(
        [
            e2ed_data["agent/front_left/img"],
            e2ed_data["agent/front/img"],
            e2ed_data["agent/front_right/img"],
        ], axis = 1
    )

    # Top subplot for front view
    ax_img = fig.add_subplot(2, 1, 1)
    ax_img.imshow(front_view)
    ax_img.axis("off")
    ax_img.set_title("Front View", fontsize=20)

    # Bottom subplot for 3D view
    ax_3d = fig.add_subplot(2, 1, 2, projection="3d", computed_zorder=False)
    ax_3d.view_init(elev=50.0, azim=-75)

    # Draw trajectory
    ax_3d.plot(traj[0:16, 0], traj[0:16, 1], zs=0, zdir='z',
               alpha=0.99, color="blue", linestyle='--', label = "past trajectory")
    ax_3d.plot(traj[16:, 0], traj[16:, 1], zs=0, zdir='z',
               alpha=0.99, color="blue", label = "future trajectory")

    ax_3d.scatter(
        traj[-1, 0], traj[-1, 1],
        zs=0, zdir='z', marker = '^',
        color="blue", s=30, zorder=5,
        label ="forward"
    )
    ax_3d.legend(loc="best")

    # Add ego-vehicle
    bbox = rotate_bbox_zaxis(car, float(0))
    bbox = shift_cuboid(float(current_pos[0]), float(current_pos[1]), bbox)
    add_cube(bbox, ax_3d, color="blue", alpha=0.1)

    ax_3d.set_zlim(bottom=0, top=5)
    ax_3d.set_aspect("equal")
    ax_3d.set_axis_off()
    ax_3d.set_facecolor("tab:grey")
    ax_3d.set_title(f"vehicle {routing[intent-1]}", fontsize=20)

    plt.tight_layout()
    plt.show()