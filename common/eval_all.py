import torch

from modules import Inertial_PoseTransformer

from eval_single import eval_dip,eval_tc,eval_cip,eval_andy,eval_virginia

def load_model(frames):
    
    model = Inertial_PoseTransformer(num_frame=frames, in_num_joints=6, out_num_joints=15, 
                                      in_chans=12, embed_dim_ratio=32, depth=4,
                                    num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0.1
                                    ,with_spatial_block = True,
                 with_spatial_pos_embed = True,
                 with_temporal_pos_embed = True,
                 with_ssms = True,
                 with_ssmt = True)
    
    return (model,'ck',30)


def eval_all():
    frames = 30
    batch_size = 2048
    device = torch.device('cuda')
    model,model_name,_ = load_model(frames=frames)
    model = model.to(device)
    weight_path = './checkpoint/ck.bin'

    eval_dip(model=model,weight_path=weight_path,batch_size=batch_size,window_size=frames,alin_to_root=True)
    eval_tc(model=model,weight_path=weight_path,batch_size=batch_size,window_size=frames,alin_to_root=True)

    eval_cip(model=model,weight_path=weight_path,batch_size=batch_size,window_size=frames)
    eval_andy(model=model,weight_path=weight_path,batch_size=batch_size,window_size=frames)

    # eval_virginia(model=model,weight_path=weight_path,batch_size=batch_size,window_size=frames)

if __name__ == '__main__':
    eval_all()

