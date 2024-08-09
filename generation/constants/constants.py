# --- type params
model_name = ""
french_hubert = False
hidden_state_index = 0 #0 to 24
resume = False
w_gan =  False
unroll_steps =  False
first_kernel_size = 0
kernel_size = 0
dropout = 1

# --- Path params
datasets = ""
datasets_properties = ""
dir_path = ""
data_path = ""
saved_path = ""
output_path = ""
evaluation_path = ""
model_path = ""

# --- Training params
n_epochs =  0
batch_size = 0
d_lr =  0
g_lr =  0
log_interval =  0
adversarial_coeff = 0
au_coeff = 0
pose_coeff = 0
eye_coeff = 0
acc_coeff = 0
fake_target = True


# --- Data params
noise_size = 0
text_size = 0 # taille des embedding de text (FlauBert)
pose_size = 0 # nombre de colonne openface pose and gaze angle
eye_size = 0 #nombre de colonne openface gaze (déja normalisé)
pose_t_size = 0 #location of the head with respect to camera
pose_r_size = 0 # Rotation is in radians around X,Y,Z axes with camera being the origin.
au_size = 0 # nombre de colonne openface AUs
prosody_size = 0 #taille des channels hubert
derivative = False

opensmile_columns = []
selected_opensmile_columns = []
selected_os_index_columns = []
openface_columns = []
number_of_opensmile_features = 25
number_visual_features = 28

#Model function
train_model = None
generate_motion = None
generate_one_file_motion = None

