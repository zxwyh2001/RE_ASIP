import sys
sys.path.append('.')

# import random
from data_load import MyDataset
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from articulate.math import *
from einops import rearrange, repeat
# from Inertial_PoseFormer import Inertial_PoseTransformer
# from models import Inertial_PoseTransformer
# from train_tran_net import TranslationNet

from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset, Subset
from articulate.model import ParametricModel
import torch

class joint_set:
    leaf = [7, 8, 12, 20, 21]
    full = list(range(1, 24))
    reduced = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    # reduced_joint_name = ['左大腿','右大腿','肚子','左膝盖','右膝盖','中间','胸部','喉咙','左胸','右胸','头部','左肩','右肩','左肘','右肘']

    

    tip_reduced_index = [1,15,4,2,16,5,6,10,7,12,11,8,13,9,14]
    ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]

    lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]

    n_leaf = len(leaf)
    n_full = len(full)
    n_reduced = len(reduced)
    n_ignored = len(ignored)

class BasePoseEvaluator:
    r"""
    Base class for evaluators that evaluate motions.
    """
    def __init__(self, official_model_file: str, rep=RotationRepresentation.ROTATION_MATRIX, use_pose_blendshape=False,
                 device=torch.device('cpu')):
        self.model = ParametricModel(official_model_file, use_pose_blendshape=use_pose_blendshape, device=device)
        self.rep = rep
        self.device = device

    def _preprocess(self, pose, shape=None, tran=None):
        pose = to_rotation_matrix(pose.to(self.device), self.rep).view(pose.shape[0], -1)
        shape = shape.to(self.device) if shape is not None else shape
        tran = tran.to(self.device) if tran is not None else tran
        return pose, shape, tran
    
class FullMotionEvaluator(BasePoseEvaluator):
    r"""
    Evaluator for full motions (pose sequences with global translations). Plenty of metrics.
    """
    def __init__(self, official_model_file: str, align_joint=None, rep=RotationRepresentation.ROTATION_MATRIX,
                 use_pose_blendshape=False, fps=60, joint_mask=None, device=torch.device('cuda')):
        r"""
        Init a full motion evaluator.

        :param official_model_file: Path to the official SMPL/MANO/SMPLH model to be loaded.
        :param align_joint: Which joint to align. (e.g. SMPLJoint.ROOT). By default the root.
        :param rep: The rotation representation used in the input poses.
        :param use_pose_blendshape: Whether to use pose blendshape or not.
        :param joint_mask: If not None, local angle error, global angle error, and joint position error
                           for these joints will be calculated additionally.
        :param fps: Motion fps, by default 60.
        :param device: torch.device, cpu or cuda.
        """
        super(FullMotionEvaluator, self).__init__(official_model_file, rep, use_pose_blendshape, device=device)
        self.align_joint = 0 if align_joint is None else align_joint.value
        self.fps = fps
        self.joint_mask = joint_mask

    def __call__(self, pose_p, pose_t, shape_p=None, shape_t=None, tran_p=None, tran_t=None):
        r"""
        Get the measured errors. The returned tensor in shape [10, 2] contains mean and std of:
          0.  Joint position error (align_joint position aligned).
          1.  Vertex position error (align_joint position aligned).
          2.  Joint local angle error (in degrees).
          3.  Joint global angle error (in degrees).
          4.  Predicted motion jerk (with global translation).
          5.  True motion jerk (with global translation).
          6.  Translation error (mean root translation error per second, using a time window size of 1s).
          7.  Masked joint position error (align_joint position aligned, zero if mask is None).
          8.  Masked joint local angle error. (in degrees, zero if mask is None).
          9.  Masked joint global angle error. (in degrees, zero if mask is None).

        :param pose_p: Predicted pose or the first pose in shape [batch_size, *] that can
                       reshape to [batch_size, num_joint, rep_dim].
        :param pose_t: True pose or the second pose in shape [batch_size, *] that can
                       reshape to [batch_size, num_joint, rep_dim].
        :param shape_p: Predicted shape that can expand to [batch_size, 10]. Use None for the mean(zero) shape.
        :param shape_t: True shape that can expand to [batch_size, 10]. Use None for the mean(zero) shape.
        :param tran_p: Predicted translations in shape [batch_size, 3]. Use None for zeros.
        :param tran_t: True translations in shape [batch_size, 3]. Use None for zeros.
        :return: Tensor in shape [10, 2] for the mean and std of all errors.
        """
        f = self.fps
        pose_local_p, shape_p, tran_p = self._preprocess(pose_p, shape_p, tran_p)
        pose_local_t, shape_t, tran_t = self._preprocess(pose_t, shape_t, tran_t)
        pose_global_p, joint_p, vertex_p = self.model.forward_kinematics(pose_local_p, shape_p, tran_p, calc_mesh=True)
        pose_global_t, joint_t, vertex_t = self.model.forward_kinematics(pose_local_t, shape_t, tran_t, calc_mesh=True)

        offset_from_p_to_t = (joint_t[:, self.align_joint] - joint_p[:, self.align_joint]).unsqueeze(1)
        ve = (vertex_p + offset_from_p_to_t - vertex_t).norm(dim=2)   # N, J
        je = (joint_p + offset_from_p_to_t - joint_t).norm(dim=2)     # N, J
        # lae = radian_to_degree(angle_between(pose_local_p, pose_local_t).view(pose_p.shape[0], -1))           # N, J
        gae = radian_to_degree(angle_between(pose_global_p, pose_global_t).view(pose_p.shape[0], -1))         # N, J
        jkp = ((joint_p[3:] - 3 * joint_p[2:-1] + 3 * joint_p[1:-2] - joint_p[:-3]) * (f ** 3)).norm(dim=2)   # N, J
        # jkt = ((joint_t[3:] - 3 * joint_t[2:-1] + 3 * joint_t[1:-2] - joint_t[:-3]) * (f ** 3)).norm(dim=2)   # N, J
        te = ((joint_p[f:, :1] - joint_p[:-f, :1]) - (joint_t[f:, :1] - joint_t[:-f, :1])).norm(dim=2)        # N, 1
        # te_10s = ((joint_p[f*10:, :1] - joint_p[:-f*10, :1]) - (joint_t[f*10:, :1] - joint_t[:-f*10, :1])).norm(dim=2)        # N, 1
        # mje = je[:, self.joint_mask] if self.joint_mask is not None else torch.zeros(1)     # N, mJ
        # mlae = lae[:, self.joint_mask] if self.joint_mask is not None else torch.zeros(1)   # N, mJ
        mgae = gae[:, self.joint_mask] if self.joint_mask is not None else torch.zeros(1)   # N, mJ

        # gaepj = gae.transpose(0,1).mean(dim=-1)

        # return torch.tensor([[je.mean(),   je.std(dim=0).mean()],
        #                      [ve.mean(),   ve.std(dim=0).mean()],
        #                      [lae.mean(),  lae.std(dim=0).mean()],
        #                      [gae.mean(),  gae.std(dim=0).mean()],
        #                      [jkp.mean(),  jkp.std(dim=0).mean()],
        #                      [jkt.mean(),  jkt.std(dim=0).mean()],
        #                      [te.mean(),   te.std(dim=0).mean()],
        #                      [mje.mean(),  mje.std(dim=0).mean()],
        #                      [mlae.mean(), mlae.std(dim=0).mean()],
        #                      [mgae.mean(), mgae.std(dim=0).mean()],
        #                      [te_10s.mean(),te_10s.std(dim=0).mean()]])
        return torch.tensor([[mgae.mean(), mgae.std(dim=0).mean()],
                             [gae.mean(),  gae.std(dim=0).mean()],
                             [je.mean(),   je.std(dim=0).mean()],
                             [ve.mean(),   ve.std(dim=0).mean()],
                             [jkp.mean(),  jkp.std(dim=0).mean()],
                             [te.mean(),   te.std(dim=0).mean()],])

class PoseEvaluator:
    def __init__(self):
        self._eval_fn = FullMotionEvaluator('./dataset/male.pkl', joint_mask=torch.tensor([1, 2, 16, 17]),device=torch.device('cuda'))

    def eval(self, pose_p, pose_t,tran_p=None, tran_t=None):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)
        pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)
        errs = self._eval_fn(pose_p, pose_t,tran_p=tran_p, tran_t=tran_t)
        # return torch.stack([errs[9], errs[3], errs[0] * 100, errs[1] * 100, errs[4] / 100, errs[6], errs[10]])
        return torch.stack([errs[0], errs[1], errs[2]*100, errs[3]*100, errs[4]/100, errs[5]])

    @staticmethod
    def print(errors):
        for i, name in enumerate(['SIP Error (deg)', 'Angular Error (deg)', 'Positional Error (cm)',
                                  'Mesh Error (cm)', 'Jitter Error (100m/s^3)','Translation error in 1s(m)', 'Translation error in 10s(m)']):
            print('%s: %.2f (+/- %.2f)' % (name, errors[i, 0], errors[i, 1]))

def add_root(y_hat,x):
    bs = y_hat.shape[0]
    y_hat = y_hat[:,-1]
    y_hat = rearrange(y_hat,'b (h w)->(b h) w',h=15)
    root_rot = x[:,-1,-1,:9].view(-1,3,3)

    root_rot_aa = rotation_matrix_to_axis_angle(root_rot)
    y_hat_aa = rotation_matrix_to_axis_angle(r6d_to_rotation_matrix(y_hat))
    y_hat_aa = rearrange(y_hat_aa,'(b c) h-> b c h',c=15)
    reduced = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    smpl_24 = torch.zeros((bs,24,3),device=y_hat.device)
    smpl_24[:,0] = root_rot_aa
    smpl_24[:,reduced] = y_hat_aa
    return smpl_24

def add_batch_root(y_hat,x):
    bs = y_hat.shape[0]

    # y_hat:(60,90)
    # x:(60,6,12)
    y_hat = rearrange(y_hat,'b (h w)->(b h) w',h=15)
    root_rot = x[:,-1,:9] #(60,9)
    root_rot = rearrange(root_rot,'b (c h)-> b c h',c=3,h=3)



    root_rot_aa = rotation_matrix_to_axis_angle(root_rot)
    y_hat_aa = rotation_matrix_to_axis_angle(r6d_to_rotation_matrix(y_hat))
    y_hat_aa = rearrange(y_hat_aa,'(b c) h-> b c h',c=15)
    reduced = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    smpl_24 = torch.zeros((bs,24,3),device=y_hat.device)
    smpl_24[:,0] = root_rot_aa
    smpl_24[:,reduced] = y_hat_aa
    return smpl_24

def eval_dip(weight_path = '',batch_size=2048,model=None,window_size=60,only_eval=True,is_save=False,alin_to_root=False):
    
    dip_eval = torch.load('./dataset/eval_set/dip_eval.pt')
    # dip_eval = torch.load('./dataset/dip/dip_eval.pt')
    dip_input = dip_eval['input']
    # print(dip_input.shape)
    dip_output = dip_eval['output']

    if only_eval:
        dip_input = dip_input[-57291:]
        dip_output = dip_output[-57291:]

    # print(dip_input.shape)
    dip_output = rearrange(dip_output,'b c h->(b c) h')
    dip_output_rm = r6d_to_rotation_matrix(dip_output)
    dip_output_rm = rearrange(dip_output_rm,'(b d) c h->b d c h',d=24)
    
    ############## ---> root ##########################
    if alin_to_root:
        glb_ori = dip_input[:,:,:9].view(-1, 6, 3, 3)
        glb_acc = dip_input[:,:,9:].view(-1, 6, 3)
        acc = torch.cat((glb_acc[:, :5] - glb_acc[:, 5:], glb_acc[:, 5:]), dim=1).bmm(glb_ori[:, -1])
        ori = torch.cat((glb_ori[:, 5:].transpose(2, 3).matmul(glb_ori[:, :5]), glb_ori[:, 5:]), dim=1)
        ori = ori.view(-1,6,9)
        dip_input = torch.cat((ori,acc),dim=-1)
    ############### ---> root ##########################
    
    # print(f'dip_input:{dip_input.shape},dip_output_rm:{dip_output_rm.shape}')

    # start = 34273 + 32589 + 33702 + 30815 + 33704 + 33355 + 30369 + 30771 + 26509
    # end =  34273 + 32589 + 33702 + 30815 + 33704 + 33355 + 30369 + 30771 + 26509
    # dip_input = dip_input[start:]
    # dip_output_rm = dip_output_rm[start:]
    
    dip_eval_dataset = MyDataset(dip_input,dip_output_rm, window_size)
    dip_eval_dataloader = DataLoader(dip_eval_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda')
    weight = torch.load(weight_path, map_location=lambda storage, loc: storage)

    # model = Inertial_PoseTransformer(num_frame=60, 
    #                     in_num_joints=6, 
    #                     out_num_joints=15, 
    #                     in_chans=12, 
    #                     embed_dim_ratio=32, 
    #                     depth=4,
    #                     num_heads=8, 
    #                     mlp_ratio=2., 
    #                     qkv_bias=True, 
    #                     qk_scale=None,
    #                     drop_path_rate=0.1).to(device)
    
    model.load_state_dict(weight['model_pos'], strict=False)

    model.eval()
    pose_ev = PoseEvaluator()
    errors = []
    preds = []
    ##### test dip_eval
    for inputs, targets in tqdm(dip_eval_dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            gt = targets[:,-1,:,:].squeeze(1)
            pred = outputs[:,-1,:].squeeze(1)
            preds.append(pred)

            pred = rearrange(pred,'b (c h)-> (b c) h',c=15)
            pred_rm = r6d_to_rotation_matrix(pred)
            
            pred_rm = rearrange(pred_rm,'(b c) h w-> b c h w',c=15)
            
            pred_rm_24 = torch.ones_like(gt)
            pred_rm_24[:,joint_set.reduced,:,:] = pred_rm
            batch_error = pose_ev.eval(pred_rm_24,gt)
            batch_error = batch_error.unsqueeze(0)
            errors.append(batch_error)
    
    errors = torch.cat(errors,dim=0)
    preds = torch.cat(preds,dim=0)
    if is_save:
        dip_input = dip_input[59:]
        smpl_pose = add_batch_root(y_hat=preds,x=dip_input)
        torch.save(smpl_pose,'dip_ours_pose.pt')
        

    #    ['SIP Error (deg)', 
    #    'Angular Error (deg)', 
    #    'Positional Error (cm)',
    #    'Mesh Error (cm)', 
    #    'Jitter Error (100m/s^3)',
    #    'Translation error in 1s(m)']

    sip_error = errors[:,0,0].mean()
    angular_error = errors[:,1,0].mean()
    pe = errors[:,2,0].mean()
    me = errors[:,3,0].mean()
    jitter = errors[:,4,0].mean()
    # te = errors[:,5,0].mean()

    file_path = './logging/dip_tc_res.txt'
    with open(file_path, 'a') as f:
        import datetime
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
        print(formatted_time,file=f)
        print('---------dip_eval_result:------------',file=f)
        print('SIP Error (deg)',sip_error,file=f)
        print('Angular Error (deg)',angular_error,file=f)
        print('Positional Error (cm)',pe,file=f)
        print('Mesh Error (cm)',me,file=f)
        print('Jitter Error (100m/s^3)',jitter/100,file=f)

    print('---------dip_eval_result:------------')
    print('SIP Error (deg)',sip_error)
    print('Angular Error (deg)',angular_error)
    print('Positional Error (cm)',pe)
    print('Mesh Error (cm)',me)
    print('Jitter Error (100m/s^3)',jitter/100)
    # print('Translation error in 1s(m)',te)

    # preds = rotation_matrix_to_axis_angle(r6d_to_rotation_matrix(preds))
    # preds = rearrange(preds,'(b c) h-> b c h',c=15)
    # preds_save = torch.zeros(preds.shape[0],24,3)
    # preds_save[:,[1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19],:] = preds.cpu()
    # torch.save(preds_save,'ours_best_dip.pt')

def eval_tc(weight_path = '',batch_size=2048,model=None,window_size=60,is_save=False,alin_to_root=False):
    
    tc_eval = torch.load('./dataset/eval_set/tc_eval.pt')
    tc_input = tc_eval['input']
    tc_output = tc_eval['output']

    # print(tc_input.shape,tc_output.shape)
    tc_output = rearrange(tc_output,'b c h->(b c) h')
    tc_output_rm = r6d_to_rotation_matrix(tc_output)
    tc_output_rm = rearrange(tc_output_rm,'(b d) c h->b d c h',d=24)
    
    ############## ---> root ##########################
    if alin_to_root:
        glb_ori = tc_input[:,:,:9].view(-1, 6, 3, 3)
        glb_acc = tc_input[:,:,9:].view(-1, 6, 3)
        acc = torch.cat((glb_acc[:, :5] - glb_acc[:, 5:], glb_acc[:, 5:]), dim=1).bmm(glb_ori[:, -1])
        ori = torch.cat((glb_ori[:, 5:].transpose(2, 3).matmul(glb_ori[:, :5]), glb_ori[:, 5:]), dim=1)
        ori = ori.view(-1,6,9)
            
        tc_input = torch.cat((ori,acc),dim=-1)
    ############## ---> root ##########################
    tc_eval_dataset = MyDataset(tc_input,tc_output_rm, window_size)
    tc_eval_dataloader = DataLoader(tc_eval_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda')
    weight = torch.load(weight_path, map_location=lambda storage, loc: storage)
    
    # model.load_state_dict(weight['model_pos'], strict=False)
    model.load_state_dict(weight, strict=False)

    model.eval()
    pose_ev = PoseEvaluator()
    errors = []
    preds = []

    ##### test tc_eval
    for inputs, targets in tqdm(tc_eval_dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            gt = targets[:,-1,:,:].squeeze(1)
            pred = outputs[:,-1,:].squeeze(1)
            preds.append(pred)
            pred = rearrange(pred,'b (c h)-> (b c) h',c=15)
            pred_rm = r6d_to_rotation_matrix(pred)
            
            pred_rm = rearrange(pred_rm,'(b c) h w-> b c h w',c=15)
            
            pred_rm_24 = torch.ones_like(gt)
            pred_rm_24[:,joint_set.reduced,:,:] = pred_rm
            batch_error = pose_ev.eval(pred_rm_24,gt)
            batch_error = batch_error.unsqueeze(0)
            errors.append(batch_error)
    
    errors = torch.cat(errors,dim=0)
    preds = torch.cat(preds,dim=0)
    if is_save:
        tc_input = tc_input[29:]
        smpl_pose = add_batch_root(y_hat=preds,x=tc_input)
        torch.save(smpl_pose,'./baseline_ssmt_tc.pt')
    
    #    ['SIP Error (deg)', 
    #    'Angular Error (deg)', 
    #    'Positional Error (cm)',
    #    'Mesh Error (cm)', 
    #    'Jitter Error (100m/s^3)',
    #    'Translation error in 1s(m)']

    sip_error = errors[:,0,0].mean()
    angular_error = errors[:,1,0].mean()
    pe = errors[:,2,0].mean()
    me = errors[:,3,0].mean()
    jitter = errors[:,4,0].mean()
    # te = errors[:,5,0].mean()

    file_path = './logging/dip_tc_res.txt'
    with open(file_path, 'a') as f:
        import datetime
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')

        print('===================================================',file=f)
        print(formatted_time,file=f)
        print('---------tc_eval_result:------------',file=f)
        print('SIP Error (deg)',sip_error,file=f)
        print('Angular Error (deg)',angular_error,file=f)
        print('Positional Error (cm)',pe,file=f)
        print('Mesh Error (cm)',me,file=f)
        print('Jitter Error (100m/s^3)',jitter/100,file=f)
    
    print('---------tc_eval_result:------------')
    print('SIP Error (deg)',sip_error)
    print('Angular Error (deg)',angular_error)
    print('Positional Error (cm)',pe)
    print('Mesh Error (cm)',me)
    print('Jitter Error (100m/s^3)',jitter/100)
    # print('Translation error in 1s(m)',te)

    # preds = rotation_matrix_to_axis_angle(r6d_to_rotation_matrix(preds))
    # preds = rearrange(preds,'(b c) h-> b c h',c=15)
    # preds_save = torch.zeros(preds.shape[0],24,3)
    # preds_save[:,[1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19],:] = preds.cpu()
    # torch.save(preds_save,'pf_ft2_tc_104_20_res.pt')

def eval_cip(model,weight_path,batch_size,window_size,is_save=False):
    
    device = torch.device('cuda')
    model = model.to(device)
    
    cip = torch.load('./dataset/eval_set/cip_eval.pt')
    cip_input = cip['input']
    cip_output = cip['output']
    
    cip_all_input = []
    cip_all_output = []
    cip_all = []

    if len(cip_input)==len(cip_output):
        length = len(cip_input)
        for i in range(length):
            imui = cip_input[i]
            posei = cip_output[i]
            di = MyDataset(input_tensor=imui,output_tensor=posei,window_size=window_size)
            cip_all.append(di)
        
        cip_eval_dataset = ConcatDataset(cip_all)
        print("cip_eval: ",len(cip_eval_dataset))
        cip_dataloader = DataLoader(cip_eval_dataset, batch_size=batch_size, shuffle=False)

    else:
        raise ValueError('frames not equal!')

    weight = torch.load(weight_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(weight['model_pos'], strict=False)

    model.eval()
    pose_ev = PoseEvaluator()
    errors = []
    preds = []

    for inputs,targets in tqdm(cip_dataloader):
        cip_all_input.append(inputs)
        cip_all_output.append(targets)

        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():

            outputs = model(inputs)
            # gt = targets[:,-1,:,:].squeeze(1)
            pred = outputs[:,-1,:]
            targets = targets[:,-1,:]
            preds.append(pred.squeeze(1))

            pred = rearrange(pred,'b (c h)-> (b c) h',c=15)
            pred_rm = r6d_to_rotation_matrix(pred)
            
            pred_rm = rearrange(pred_rm,'(b c) h w-> b c h w',c=15)
            
            pred_rm_24 = torch.ones_like(targets)
            pred_rm_24[:,joint_set.reduced,:,:] = pred_rm
            batch_error = pose_ev.eval(pred_rm_24,targets)
            batch_error = batch_error.unsqueeze(0)
            errors.append(batch_error)
    
    errors = torch.cat(errors,dim=0)
    preds = torch.cat(preds,dim=0)
    
    sip_error = errors[:,0,0].mean()
    angular_error = errors[:,1,0].mean()
    pe = errors[:,2,0].mean()
    me = errors[:,3,0].mean()
    jitter = errors[:,4,0].mean()
    # te = errors[:,5,0].mean()

    print('---------cip_eval_result:------------')
    print('SIP Error (deg)',sip_error)
    print('Angular Error (deg)',angular_error)
    print('Positional Error (cm)',pe)
    print('Mesh Error (cm)',me)
    print('Jitter Error (100m/s^3)',jitter/100)
    # print('Translation error in 1s(m)',te)

    if is_save:
        print(preds.shape)
        cip_all_input = torch.cat(cip_all_input,dim=0)
        cip_all_input = cip_all_input[:,-1,]

        cip_all_output = torch.cat(cip_all_output,dim=0)
        cip_all_output = cip_all_output[:,-1,]

        print(cip_all_input.shape)
        print(cip_all_output.shape)

        # torch.save(cip_all_output,'./cip_moda.pt')

        # cip_all_input = cip_all_input[29:]
        # print(cip_all_input.shape)
        smpl_pose = add_batch_root(y_hat=preds,x=cip_all_input)
        print(smpl_pose.shape)
        torch.save(smpl_pose,'./cip_moda.pt')

def eval_andy(model,weight_path,batch_size,window_size):
    
    device = torch.device('cuda')
    model = model.to(device)
    
    andy_eval = torch.load('./dataset/eval_set/andy_eval.pt')
    andy_eval_input = andy_eval['input']
    andy_eval_output = andy_eval['output']

    cip_all = []

    if len(andy_eval_input)==len(andy_eval_output):
        length = len(andy_eval_input)
        for i in range(length):
            imui = andy_eval_input[i]
            posei = andy_eval_output[i]
            di = MyDataset(input_tensor=imui,output_tensor=posei,window_size=window_size)
            cip_all.append(di)
        
        cip_train_dataset = ConcatDataset(cip_all)
        print("andy_eval: ",len(cip_train_dataset))
        cip_dataloader = DataLoader(cip_train_dataset, batch_size=batch_size, shuffle=False)

    else:
        raise ValueError('frames not equal!')

    weight = torch.load(weight_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(weight['model_pos'], strict=False)

    model.eval()
    pose_ev = PoseEvaluator()
    errors = []
    preds = []

    for inputs,targets in tqdm(cip_dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            # gt = targets[:,-1,:,:].squeeze(1)
            pred = outputs[:,-1,:]
            targets = targets[:,-1,:]
            bs = pred.shape[0]
            # preds.append(pred)

            pred = rearrange(pred,'b (c h)-> (b c) h',c=15)
            targets = rearrange(targets,'b (c h)-> (b c) h',c=15)
            
            pred_rm = r6d_to_rotation_matrix(pred)
            targets_rm = r6d_to_rotation_matrix(targets)
            
            pred_rm = rearrange(pred_rm,'(b c) h w-> b c h w',c=15)
            targets_rm = rearrange(targets_rm,'(b c) h w-> b c h w',c=15)
            
            pred_rm_24 = torch.ones(bs,24,3,3).to(device)
            targets_rm_24 = torch.ones(bs,24,3,3).to(device)
            
            pred_rm_24[:,joint_set.reduced,:,:] = pred_rm
            targets_rm_24[:,joint_set.reduced,:,:] = targets_rm
            batch_error = pose_ev.eval(pred_rm_24,targets_rm_24)
            batch_error = batch_error.unsqueeze(0)
            errors.append(batch_error)
    
    errors = torch.cat(errors,dim=0)
    # preds = torch.cat(preds,dim=0)
    
    sip_error = errors[:,0,0].mean()
    angular_error = errors[:,1,0].mean()
    pe = errors[:,2,0].mean()
    me = errors[:,3,0].mean()
    jitter = errors[:,4,0].mean()
    # te = errors[:,5,0].mean()

    print('---------andy_eval_result:------------')
    print('SIP Error (deg)',sip_error)
    print('Angular Error (deg)',angular_error)
    print('Positional Error (cm)',pe)
    print('Mesh Error (cm)',me)
    print('Jitter Error (100m/s^3)',jitter/100)
    # print('Translation error in 1s(m)',te)

