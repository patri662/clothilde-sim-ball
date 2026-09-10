import sys,os
notebook_dir = os.getcwd()  
parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))  # Get the parent directory of the current working directory (python_code folder)
print(parent_dir)
sys.path.append(parent_dir)
# from implementation.Cloth import Cloth 
from implementation.Cloth_Solid import Cloth # need to call new code, which is in a new file!
from implementation.utils import createRectangularMesh, duplicate_node_pairs, createSolidMesh, barTemplate, cubeSurfaceTemplate
import time
import numpy as np
import polyscope as ps

na = 30; nb = 18
X, T = createRectangularMesh(a = 0.9,b = 0.5, na = na, nb = nb, h = 0.15)
X[:,2] += 0.4; #adjust height

clothilde = Cloth(X, T)

# # create a bar as a solid
# # chose n_segments and a node radius that fills the whole bar
# # if spaces throughout the bar, the free-node spaces don't collide and pass through the cloth
# template, edges = barTemplate(length=0.7, axis=0, n_segments=20) # local shape: 0.7m apart, with 20 intermediate nodes
# positions, edges, faces = createSolidMesh([0,0,0.6], template, edges)  # place it at (0,0,0.6)
# clothilde.solid = clothilde.addSolid(positions, rad=0.02, mass=1, friction=0.3, edges=edges) # rad = rad of each node, so reduce rad is solid nodes vert close to each other colliding

# create a cube as solid
# better to choose smaller rad for solid nodes (so doesn't look like solid far away from cloth), but more quantity of nodes in template (with n)
template, edges, faces = cubeSurfaceTemplate(side=0.2, n=8) # side=0.05m, 3x3 grid per face, 6 faces, 54 nodes -> change n=2 for 2x2 grid per face, 6 faces, 24 nodes and n=4 for 4x4 grid per face, 6 faces, 96 nodes, etc.
positions, edges, faces = createSolidMesh([0.3,0,0.6], template, edges, faces)  # place it at (0,0,0.6)
clothilde.solid = clothilde.addSolid(positions, rad=0.01, mass=0.6, friction=0.2, edges=edges, faces=faces, name="Cube") # rad = rad of each node, so reduce rad is solid nodes vert close to each other colliding

print(f'Solid position: {clothilde.solid.positions}')   # position
print(f'Solid speed: {clothilde.solid.velocities}')      # velocity
print(f'Solid radius: {clothilde.solid.rad}')    # radius
print(f'Solid friction: {clothilde.solid.mu_b}')     # friction
print(f'Solid position history: {clothilde.solid.history_pos}')  # full trajectory

# solver parameters
dt = 1/60 #frame rate
#dt = clothilde.estimateTimeStep(L=0.9)
tol = 0.009 # up to 0.75% of relative error in constraint satisfaction to stop iterations

#physical parameters
rho = 0.1 #cloth density
delta = 0.1 # aerodynamics parameter: between 0 and rho
kappa = 0.35*1e-4 # stifness or bending resistance
kappa_bnd = 0.015*1e-4 # stifness or bending resistance

alpha = 0.3 #damping of oscillations
shr = 5*1e-4 #allowed shearing resistance
strh = 0.005*1e-4 #allowed stretching resistance
mu_f = 0.45 #friction with the floor
mu_s = 0.4 #friction with the cloth itself
thck = 1.1 #size of the balls
sub_steps = 12 #number of intermidate steps between each dt

clothilde.setSimulatorParameters(dt=dt,tol=tol,sub_steps = sub_steps,
                                rho=rho,delta=delta,kappa=kappa,kappa_bnd=kappa_bnd,shr=shr,
                                str=strh,alpha=alpha,mu_f=mu_f,mu_s=mu_s,
                                thck=thck)

clothilde.preparePolyscope()
# clothilde.plotMesh()
clothilde.solid.plotMesh()

# take the 4 corners and maintain them still to have an horizontal cloth
inds_ctr = [0,na-1, (na*nb)-na, (na*nb)-1]
tf = int(2.5/dt)
u = X[inds_ctr]
for i in range(tf):
    print(i/tf)
    clothilde.simulate(u = u, control = inds_ctr)

# release two bottom corners and let the cloth fall 
inds_ctr = [0, na-1]
# inds_ctr = []
tf = int(0*2.5/dt)
u = X[inds_ctr]
for i in range(tf):
    print(i/tf)
    clothilde.simulate(u = u, control = inds_ctr)

print('Average iterations',clothilde.total_iters/(len(clothilde.history_pos)-1))

# Save video of simulation
def save_frames_ps(history_pos, solid, Am, label, folder="frames", step=4):
    """
    Pass clothilde.solid directly, we need its label, its full position history, 
    and whether it has `faces` or is edges-only. 
    Same three things Cloth.makeMovie()'s own per-frame
    callback updates for live playback; this just does it into a screenshot
    loop instead.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(folder):
        folder = os.path.join(script_dir, folder)
    os.makedirs(folder, exist_ok=True)
    n_frames = len(history_pos)
    print(f"Saving {n_frames//step} frames to {folder}/")
 
    for i, frame_idx in enumerate(range(0, n_frames, step)):
        # update cloth
        phi_mat = history_pos[frame_idx]
        phi_all = Am @ phi_mat
        ps.get_surface_mesh(label).update_vertex_positions(phi_all)
 
        # update solid: point cloud always, surface mesh only if it has faces
        # (e.g. cubeSurfaceTemplate), curve network always (the wireframe/tube)
        solid_pos = solid.history_pos[frame_idx]
        ps.get_point_cloud(solid.label).update_point_positions(solid_pos)
        if solid.faces is not None:
            ps.get_surface_mesh(solid.label + "_surface").update_vertex_positions(solid_pos)
        else:
            ps.get_curve_network(solid.label + "_edges").update_node_positions(solid_pos)
 
        ps.screenshot(os.path.join(folder, f"frame_{i:04d}.png"), transparent_bg=False)
 
        if i % 50 == 0:
            print(f"Saved frame {i} / {n_frames//step}")

import subprocess
import shutil

def create_video_from_frames(frames_folder, output_video, framerate=20, cleanup = True):
    """
    Create a video from a sequence of frames using ffmpeg.
    
    Args:
        frames_folder: Path to folder containing frame_XXXX.png files
        output_video: Output video filename (e.g., "experiment.mp4")
        framerate: Framerate for the video (default: 20)
        cleanup: If True, delete the frames folder after video creation (default: True)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(frames_folder):
        frames_folder = os.path.join(script_dir, frames_folder)
    if not os.path.isabs(output_video):
        output_video = os.path.join(script_dir, output_video)

    try:
        cmd = [
            "ffmpeg",
            "-framerate", str(framerate),
            "-i", f"{frames_folder}/frame_%04d.png",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-y",  # overwrite output file without asking
            output_video
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Video saved: {output_video}")

        # Clean up frames folder if requested
        if cleanup:
            shutil.rmtree(frames_folder)
            print(f"Removed frames folder: {frames_folder}")

    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e.stderr.decode()}")
    except FileNotFoundError:
        print("ffmpeg not found. Install it using: brew install ffmpeg")
    except Exception as e:
        print(f"Error during cleanup: {e}")

# Save frames and create video
frames_folder = f"Bag-Solid Manipulation Frames Delete"
os.makedirs(frames_folder, exist_ok=True)

# Fix camera view
ps.reset_camera_to_home_view()
ps.set_view_projection_mode("perspective")
ps.set_window_size(1500, 1000)

intrinsics = ps.CameraIntrinsics(fov_vertical_deg=45., fov_horizontal_deg=45.)

angle = np.radians(80)
radius = 3
height = 1

# Compute camera position
cam_x = radius * np.sin(angle)   # horizontal offset
cam_z = height                    # vertical offset (low, close to table)
cam_y = radius * np.cos(angle)    # forward/back offset

# Look at origin
look_dir = (-cam_x, -cam_y, -cam_z)
up_dir = (0, 0, 1) # with the mesh rotated, now the Y-axis is the original Z-axis, so we set up to be along global Z

extrinsics = ps.CameraExtrinsics(root=(cam_x, cam_y, cam_z), look_dir=look_dir, up_dir=up_dir)

ps.set_view_camera_parameters(ps.CameraParameters(intrinsics, extrinsics))

# edges configuration
label = clothilde.label
mesh = ps.get_surface_mesh(label)

mesh.set_color((0.2, 0.5, 0.8)) # RGB color of the mesh, blue for all meshes/cases (if not determined, each mesh for each new experiment will have different colors when polyscope registers them)
mesh.set_edge_color((0.0, 0.0, 0.0))  # RGB color of the edges, black for edges
mesh.set_edge_width(0.4)             # thickness of the edges

# change view here and will save as here!
clothilde.makeMovie(1,True,0) # True/False to repeat simulation when finishes

save_frames_ps(clothilde.history_pos, clothilde.solid, clothilde.Am, 
                clothilde.label, folder=frames_folder, step=10)
create_video_from_frames(frames_folder, "Bag-Solid_Cube-Release.mp4", framerate=15, cleanup=True) 