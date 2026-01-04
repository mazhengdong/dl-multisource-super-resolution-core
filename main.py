import numpy as np 
from util import *
import sys, os, time, argparse, shutil, h5py, torch
from scipy.stats import pearsonr
import multiprocessing as mp
from models import discModel
from models import encodedGenerator
from data import bkgdGen, gen_train_batch_bg, get1batch4test
import pandas as pd
import os

parser = argparse.ArgumentParser(description='encode sinogram image.')
# 创建一个 ArgumentParser 对象 parser，并为其提供一个描述信息 'encode sinogram image.'，这个描述会在用户使用 -h 选项查看帮助信息时显示。
parser.add_argument('-gpus',   type=str, default="", help='list of visiable GPUs')
# 定义一个名为 -gpus 的命令行参数，其类型为字符串 str，默认值为空字符串 ""。该参数用于指定可见的 GPU 列表，例如 "0,1" 表示使用编号为 0 和 1 的 GPU。
parser.add_argument('-expName',type=str, default="debug", help='Experiment name')
# 定义一个名为 -expName 的命令行参数，类型为字符串 str，默认值为 "debug"。此参数用于指定实验名称，方便用户区分不同的实验设置和结果。
parser.add_argument('-depth',  type=int, default=1, help='input depth')
# 定义一个名为 -depth 的命令行参数，类型为整数 int，默认值为 1。它可能表示输入数据的深度，具体含义取决于项目的上下文，例如在处理图像数据时，可能表示图像的通道数或其他与数据维度相关的概念。
parser.add_argument('-lr',     type=float, default=3e-4, help='learning rate')
# 定义一个名为 -lr 的命令行参数，类型为浮点数 float，默认值为 3e - 4（即 0.0003）。这通常表示学习率，是深度学习训练过程中的一个重要超参数，用于控制模型参数更新的步长。
parser.add_argument('-maxep',  type=int, default=1, help='max training epoches')
# 定义一个名为 -maxep 的命令行参数，类型为整数 int，默认值为 8000。它表示最大训练轮数，即模型在训练过程中遍历整个训练数据集的次数上限。
'''
在这里进行了改动
'''
parser.add_argument('-warmup', type=int, default=100, help='warmup training epoches')
# 定义一个名为 -warmup 的命令行参数，类型为整数 int，默认值为 100。这可能表示热身训练轮数，在某些训练策略中，在开始正常训练之前，会先进行一定轮数的热身训练，以帮助模型更好地收敛。
parser.add_argument('-mbsize', type=int, default=8, help='mini batch size')
'''
在这里进行了改动
'''
# 定义一个名为 -mbsize 的命令行参数，类型为整数 int，默认值为 64。它表示小批量大小，即在每次训练迭代中，模型使用的样本数量。较大的批量大小可以利用更多的计算资源，但可能会导致内存消耗增加和收敛速度变慢；较小的批量大小则相反。
parser.add_argument('-mdlsz',  type=int, default=256, help='channels of the 1st box')
# 定义一个名为 -mdlsz 的命令行参数，类型为整数 int，默认值为 256。它可能表示模型中第一个模块（box）的通道数，在深度学习模型（如卷积神经网络）中，通道数是一个重要的参数，影响模型的特征提取能力。
parser.add_argument('-print',  type=str2bool, default=False, help='1:print to terminal; 0: redirect to file')
# 定义一个名为 -print 的命令行参数，其类型为自定义的 str2bool（可能是在其他地方定义的将字符串转换为布尔值的函数），默认值为 False。该参数用于控制输出方式，1 表示将输出打印到终端，0 表示将输出重定向到文件。
parser.add_argument('-logtrs', type=str2bool, default=False, help='log transform')
# 定义一个名为 -logtrs 的命令行参数，类型为 str2bool，默认值为 False。它可能表示是否对数据进行对数变换，这种变换在处理某些数据分布时可能会很有用，例如在处理具有较大动态范围的数据时。
parser.add_argument('-sam',    type=str2bool, default=True, help='apply spatial attention')
# 定义一个名为 -sam 的命令行参数，类型为 str2bool，默认值为 True。它用于决定是否应用空间注意力机制，空间注意力机制可以帮助模型更关注输入数据在空间维度上的重要区域。
parser.add_argument('-cam',    type=str2bool, default=True, help='apply channel attention')
# 定义一个名为 -cam 的命令行参数，类型为 str2bool，默认值为 True。它用于决定是否应用通道注意力机制，通道注意力机制可以使模型在通道维度上自动学习不同通道的重要性。
parser.add_argument('-cvars',  type=str2list, default='lrad:prcp:pres:shum:srad:temp:wind', help='vars as condition')
# parser.add_argument('-cvars',  type=str2list, default='lrad:prcp:pres:shum:srad:temp:wind', help='vars as condition')
# 定义一个名为 -cvars 的命令行参数，类型为自定义的 str2list（可能是将字符串按特定分隔符转换为列表的函数），默认值为 'T2:IWV:SLP'。它可能表示作为条件的变量列表，具体取决于项目的上下文，例如在某些模型中，这些变量可能作为额外的输入条件影响模型的输出。
parser.add_argument('-wmse',   type=float, default=5, help='weight of content loss for G loss')
# 定义一个名为 -wmse 的命令行参数，类型为浮点数 float，默认值为 5。它可能表示生成器（G）损失中内容损失（如均方误差损失，MSE loss）的权重，用于在计算生成器损失时调整内容损失项的重要性。


args, unparsed = parser.parse_known_args()
if len(unparsed) > 0:
    print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
    exit(0)



if len(args.gpus) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
torch_devs = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("log to %s, log Trans: %s, mb: %d, mdlsz: %d, CAM: %s, SAM: %s, cvars:%s, wmse:%.1f" % ('Terminal' if args.print else 'file', args.logtrs, args.mbsize, args.mdlsz,args.cam, args.sam, ','.join(args.cvars), args.wmse))



itr_out_dir = args.expName + '-itrOut'
if os.path.isdir(itr_out_dir):
    shutil.rmtree(itr_out_dir)
os.mkdir(itr_out_dir)
if args.print == 0:
    sys.stdout = open(os.path.join(itr_out_dir, 'iter-prints.log'), 'w')



def main(args):
    mb_size = args.mbsize
    in_depth = args.depth

    gene_model = encodedGenerator(in_ch=in_depth, ncvar=len(args.cvars), use_ele=True, sam=args.sam, cam=args.cam,
                                  stage_chs=[args.mdlsz // 2 ** _d for _d in range(3)])
    disc_model = discModel()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            gene_model = torch.nn.DataParallel(gene_model)
            disc_model = torch.nn.DataParallel(disc_model)
        gene_model = gene_model.to(torch_devs)
        disc_model = disc_model.to(torch_devs)

    gene_criterion = torch.nn.L1Loss()
    disc_criterion = torch.nn.BCELoss()
    gene_optimizer = torch.optim.Adam(gene_model.parameters(), lr=args.lr)
    disc_optimizer = torch.optim.Adam(disc_model.parameters(), lr=args.lr)

    lrdecay_lambda = lambda epoch: cosine_decay(epoch, warmup=args.warmup, max_epoch=args.maxep)
    gene_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(gene_optimizer, lr_lambda=lrdecay_lambda)
    disc_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(disc_optimizer, lr_lambda=lrdecay_lambda)

    mb_data_iter = bkgdGen(data_generator=gen_train_batch_bg(mb_size=mb_size, in_depth=in_depth, dev=torch_devs, trans=args.logtrs, cvars=args.cvars), max_prefetch=mb_size*4)

    ele_12km = h5py.File(r'E:\文章2\实验\2013\NDVI\elevation_12km_resized.hdf5', "r")["elevation"]
    ele_12km = np.array([ele_12km] * mb_size)
    ele_12km = np.expand_dims(ele_12km, 1)
    ele_12km = torch.from_numpy(ele_12km).to(torch_devs)

    # get disc out size and create label
    dsc_out_size= (mb_size, 1, 4, 8)
    true_label  = torch.ones (dsc_out_size)
    false_label = torch.zeros(dsc_out_size)
    disc_label  = torch.cat((true_label, false_label), dim=0).to(torch_devs)

    for epoch in range(args.maxep+1):
        time_it_st = time.time()
        X_mb, cvars, y_mb = mb_data_iter.next() # with prefetch

        gene_optimizer.zero_grad()
        pred = gene_model.forward(X_mb, cvars, ele_12km)
        with torch.no_grad():
            advs_loss = 0 - disc_model.forward(pred).mean().log() # adv loss
        cont_loss = gene_criterion(pred, y_mb) # content loss
        gene_loss = args.wmse * cont_loss + advs_loss
        gene_loss.backward()
        gene_optimizer.step()
        gene_lr_scheduler.step()

        disc_optimizer.zero_grad()
        disc_mb   = torch.cat((y_mb, pred.detach()), dim=0)
        disc_pred = disc_model.forward(disc_mb)
        disc_loss = disc_criterion(disc_pred, disc_label) / 2 # slows down the rate relative to G
        disc_loss.backward()
        disc_optimizer.step()
        disc_lr_scheduler.step()

        itr_prints = '[Info] @ %.1f Epoch: %05d, gloss: %.2f = (Cont:%.2f + Adv:%.2f), dloss: %.2f, elapse: %.2fs/itr, lr: %.5f' % (time.time(), epoch, gene_loss.detach().cpu().numpy(), cont_loss.detach().cpu().numpy(), advs_loss.detach().cpu().numpy(), disc_loss.detach().cpu().numpy(), (time.time() - time_it_st), gene_optimizer.param_groups[0]['lr'])
        print(itr_prints)

        if epoch % (500) == 0:
            if epoch == 0:
                # X222, cv222, y222 = get1batch4test(in_depth=in_depth, idx=range(args.mbsize), dev=torch_devs,trans=args.logtrs, cvars=args.cvars)
                X222, cv222, y222, original_days = get1batch4test(in_depth=in_depth, idx=range(args.mbsize),dev=torch_devs, trans=args.logtrs, cvars=args.cvars)
                save2img_rgb(X222[0, in_depth - 1, :, :].cpu(), '%s/low-res.png' % (itr_out_dir))
                pd.DataFrame(X222[0, in_depth - 1, :, :].cpu().numpy()).to_csv(f'{itr_out_dir}/low-res.csv',index=False)

                true_img = y222.cpu().numpy()
                if args.logtrs: true_img = np.exp(true_img) - 1  # transform back
                save2img_rgb(true_img[0, 0, :, :], '%s/high-res.png' % (itr_out_dir))
                pd.DataFrame(true_img[0, 0, :, :]).to_csv(f'{itr_out_dir}/high-res.csv', index=False)

            with torch.no_grad():
                pred_img = gene_model.forward(X222, cv222, ele_12km).cpu().numpy()
                if args.logtrs: pred_img = np.exp(pred_img) - 1  # transform back
                mse = np.mean((true_img - pred_img) ** 2)
                cc_avg = np.mean(
                    [pearsonr(pred_img[i].flatten(), true_img[i].flatten())[0] for i in range(pred_img.shape[0])])
            print(
                '[Validation] @ Epoch: %05d MSE: %.4f, CC:%.3f of %d samples' % (epoch, mse, cc_avg, pred_img.shape[0]))

            # 保存所有生成的预测图片
            for i in range(args.mbsize):
                day = original_days[i]  # 获取实际日期编号
                save2img_rgb(pred_img[i, 0, :, :], '%s/pred_img_day%03d_epoch%05d.png' % (itr_out_dir, day, epoch))
                pd.DataFrame(pred_img[i, 0, :, :]).to_csv(f'{itr_out_dir}/pred_img_day{day:03d}_epoch{epoch:05d}.csv',index=False)

            if torch.cuda.device_count() > 1:
                torch.save(gene_model.module.state_dict(), "%s/mdl-it%05d.pth" % (itr_out_dir, epoch))
            else:
                torch.save(gene_model.state_dict(), "%s/mdl-it%05d.pth" % (itr_out_dir, epoch))

        sys.stdout.flush()

if __name__ == '__main__':
    main(args)
