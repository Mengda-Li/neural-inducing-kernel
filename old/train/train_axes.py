print('importing packages...')
import os
import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from PIL import Image
print('importing torch...')
import torch
from torch.utils.tensorboard import SummaryWriter
import torch._dynamo
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms.functional import to_tensor
print('importing local...')
from dataset import TrainMIT5KDataset
from splines import AxisSimplestSpline, TPS2RGBSpline, TPS2RGBSplineXY, SimplestSpline, AxisPerImageSimplestSpline
from config import DATASET_DIR, DEVICE
from ptcolor import squared_deltaE94, rgb2lab
print('ended imports, starting...')

def perimage_validate_image(backbone, spline, val_img, logdir):
    if not os.path.exists(logdir):
        Path(logdir).mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        params_tensor_batch = backbone(val_img)
        params = spline.get_params(params_tensor_batch)  # {'ys':...}
        A = params['A']  # (3, n_axis)
        ys = params['ys']  # (n_axis, n_knots)

        print("A", params['A'], "A".shape, type(A))

        maxes = (A*(A>0)).sum(dim=0) # np.sum(np.abs(A), axis=0)
        mins = (A*(A<0)).sum(dim=0) #np.sum(A*(A<0), axis=0)
        nch, nax = A.shape
        xs = np.linspace(mins[0], maxes[0], ys.shape[1]+2)
        plt.figure()
        plt.plot(xs, [0]+list(ys[0])+[1], c='r')
        plt.plot(xs, [0]+list(ys[1])+[1], c='g')
        plt.plot(xs, [0]+list(ys[2])+[1], c='b')
        for i in range(nax):
            xs = np.linspace(mins[i], maxes[i], ys.shape[1]+2)
            plt.plot(xs, [mins[i]]+list(ys[i])+[maxes[i]], label=f'axis {i}')
        plt.legend()
        plt.savefig(logdir / 'params.png')
        plt.close()
        out_batch = spline(val_img, params_tensor_batch)  # (1, 3, H, W)
        outimg = outimg.resize((W, H))
        outimg.save(logdir / 'out.jpg')
 
        ax = plt.figure().add_subplot(projection='3d')
        origin = np.zeros((3,))
        ax.quiver(origin, origin, origin, A[0,:], A[1,:], A[2,:])
        plt.savefig(logdir / "directions.png")
        

        # ys = np.array(ys[0].cpu())
        # xs = np.linspace(0, 1, ys.shape[1]+2)
        #mins = (A * (A < 0)).sum(dim=0)  # (n_axis,)
        #maxs = (A * (A > 0)).sum(dim=0)

        # plt.figure()
        # for axind, axes in enumerate(A.T):
        #     plt.plot(
        #         torch.linspace(mins[axind].item(), maxs[axind].item(), spline.n_knots+2),
        #         torch.cat((mins[axind][None], ys[0, axind], maxs[axind][None])).cpu(),
        #         label=axes
        #         )
        # plt.legend()
        # plt.savefig(logdir / f'params.png')
        # plt.close()

        # plt.figure()
        # plt.plot(xs, [0]+list(ys[0])+[1], c='r')
        # plt.plot(xs, [0]+list(ys[1])+[1], c='g')
        # plt.plot(xs, [0]+list(ys[2])+[1], c='b')
        # plt.savefig(logdir / 'params.png')
        # plt.close()

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # for axind, axes in enumerate(A.T.cpu()):
        #     ax.plot([0, axes[0]], [0, axes[1]], [0, axes[2]], label=[round(e,2) for e in list(axes.numpy())])
        # # set the axis labels
        # ax.set_xlabel('R')
        # ax.set_ylabel('G')
        # ax.set_zlabel('B')
        # plt.legend(bbox_to_anchor=(1.0, 0.5))
        # plt.savefig(logdir / f'axis1.png', bbox_inches='tight')
        # # rotate the view
        # ax.view_init(azim=-45, elev=45)
        # plt.savefig(logdir / f'axis2.png', bbox_inches='tight')
        # ax.view_init(azim=-15, elev=60)
        # plt.savefig(logdir / f'axis3.png', bbox_inches='tight')
        # plt.close()

        # out_batch = spline(val_img, params)
        # out = np.clip(out_batch[0].permute(1, 2, 0).cpu().numpy(), 0, 1)  # (H, W, 3)
        # outimg = Image.fromarray((out * 255).astype("uint8"))
        # outimg = outimg.resize((W, H))
        # outimg.save(logdir / 'out.jpg')



def validate_image(backbone, A, spline, val_img, logdir):
    if not os.path.exists(logdir):
        Path(logdir).mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        params = {'A':A}
        params_tensor_batch = backbone(val_img)
        ys = spline.get_params_ys(params_tensor_batch)['ys']  # {'ys':...}
        params['ys'] = ys
 

        # ys = np.array(ys[0].cpu())
        # xs = np.linspace(0, 1, ys.shape[1]+2)
        mins = (A * (A < 0)).sum(dim=0)  # (n_axis,)
        maxs = (A * (A > 0)).sum(dim=0)

        plt.figure()
        for axind, axes in enumerate(A.T):
            plt.plot(
                torch.linspace(mins[axind], maxs[axind], spline.n_knots+2),
                torch.cat((mins[axind][None], ys[0, axind], maxs[axind][None])).cpu(),
                label=axes
                )
        plt.legend()
        plt.savefig(logdir / f'params.png')
        plt.close()

        # plt.figure()
        # plt.plot(xs, [0]+list(ys[0])+[1], c='r')
        # plt.plot(xs, [0]+list(ys[1])+[1], c='g')
        # plt.plot(xs, [0]+list(ys[2])+[1], c='b')
        # plt.savefig(logdir / 'params.png')
        # plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for axind, axes in enumerate(A.T.cpu()):
            ax.plot([0, axes[0]], [0, axes[1]], [0, axes[2]], label=[round(e,2) for e in list(axes.numpy())])
        # set the axis labels
        ax.set_xlabel('R')
        ax.set_ylabel('G')
        ax.set_zlabel('B')
        plt.legend(bbox_to_anchor=(1.0, 0.5))
        plt.savefig(logdir / f'axis1.png', bbox_inches='tight')
        # rotate the view
        ax.view_init(azim=-45, elev=45)
        plt.savefig(logdir / f'axis2.png', bbox_inches='tight')
        ax.view_init(azim=-15, elev=60)
        plt.savefig(logdir / f'axis3.png', bbox_inches='tight')
        plt.close()

        out_batch = spline(val_img, params)
        out = np.clip(out_batch[0].permute(1, 2, 0).cpu().numpy(), 0, 1)  # (H, W, 3)
        outimg = Image.fromarray((out * 255).astype("uint8"))
        outimg = outimg.resize((W, H))
        outimg.save(logdir / 'out.jpg')

def fit(
    backbone,
    spline,
    dataloader,
    optimizer,
    scheduler,
    loss_fn,
    n_epochs=500,
    verbose=True,
    profiler=None,
    exp_avg=False,
):
    logger = SummaryWriter()
    logdir = Path(logger.get_logdir())
    val_img = Image.open(Path(DATASET_DIR) / "train" / "raw" / "004999.jpg")
    H, W = val_img.height, val_img.width
    val_img = val_img.resize((448,448))
    val_img = to_tensor(val_img).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)

    running_loss = 0
    if not exp_avg:
        running_count = 0
    for epoch_idx in range(n_epochs):
        with tqdm.tqdm(total=len(dataloader), desc=f"Epoch {epoch_idx}") as pbar:
            for i, (raw_batch, target_batch) in enumerate(dataloader):
                raw_batch, target_batch = raw_batch.to(DEVICE), target_batch.to(DEVICE)
                params_tensor_batch = backbone(raw_batch)
                params = spline.get_params(params_tensor_batch)
                A = params['A']
                Areg_loss = 0.1 * ((torch.norm(A, dim=0) - 1)**2).sum()  # unit norm
                params['A'] = A / torch.norm(A, dim=0, keepdim=True)
                out_batch = spline(raw_batch, params)
                imgdiffloss = loss_fn(out_batch, target_batch)
                loss = imgdiffloss + Areg_loss
                # loss += (-params_tensor_batch > 0).sum() * 10
                # loss += (params_tensor_batch-1 > 0).sum() * 1
                loss.backward()
                optimizer.step()
                # scheduler.step()
                scheduler.step(loss)

                logger.add_scalar("train/loss", loss, epoch_idx * len(dataloader) + i)
                logger.add_scalar("train/Aregloss", Areg_loss, epoch_idx * len(dataloader) + i)
                logger.add_scalar("train/imgdiffloss", imgdiffloss, epoch_idx * len(dataloader) + i)
                if exp_avg:
                    # exponential moving average
                    if running_loss == 0:
                        running_loss = imgdiffloss.item()
                    else:
                        running_loss = 0.9 * running_loss + 0.1 * imgdiffloss.item()
                else:
                    running_count += 1
                    running_loss = running_loss * (running_count-1) / running_count + imgdiffloss.item() / running_count
                pbar.update(1)
                pbar.set_postfix({"loss": running_loss, "batch": i})



                # if i % 6 == 0:  # dev
                #     validate_image(backbone, A, spline, val_img, logdir)


                if profiler:
                    # save profiler stats
                    profiler.disable()
                    profiler.dump_stats(logdir / f"profiler.prof")
                    profiler.enable()

            validate_image(backbone, A, spline, val_img, logdir)
            torch.save(backbone.state_dict(), logdir / f"backbone_{epoch_idx}.pth")
    return backbone

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    SEED = 0
    splineclass = AxisPerImageSimplestSpline
    per_image = True
    lr = 1e-5
    n_knots = 10
    batch_size = 40
    n_epochs = 1000
    initckpt = "backbone_axisperimage.pth"
    inittol = 0.0005
    reset = True#False
    import cProfile

    seed_everything(SEED)
    # initialize profiler
    pr = cProfile.Profile()
    pr.enable()

    print('init spline...')
    A = torch.tensor([[1,0,0], [0,1,0], [0,0,1]
                      , [1,1,1]
                      ]).float()
    A = (A / torch.norm(A, dim=1, keepdim=True)).T
    A = A.to(DEVICE)
    print("A", A.shape)
    A.requires_grad = not per_image
    spline = splineclass(n_knots=n_knots, n_axis=A.shape[1]).to(DEVICE)
    n_params = spline.get_n_params()  # predict only ys not A
    torch._dynamo.config.verbose = True
    spline = torch.compile(spline, mode="reduce-overhead", disable=True)

    print('init backbone...')
    backbone = torchvision.models.mobilenet_v3_small(num_classes=n_params).to(DEVICE)
    # current_model_dict = backbone.state_dict()
    # loaded_state_dict = torch.load("backbone_23.pth", map_location=DEVICE)
    # new_state_dict={k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), loaded_state_dict.values())}
    # backbone.load_state_dict(new_state_dict, strict=False)
    # net.fc = torch.nn.Linear(512, n_params)

    print('init dataset...')
    dataset = TrainMIT5KDataset(datadir=DATASET_DIR)
    assert len(dataset) > 0, "dataset is empty"
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True
    )
    
    print('init weights...')
    if os.path.isfile(initckpt) and not reset:
        state_dict = torch.load(initckpt, map_location=DEVICE)
        backbone.load_state_dict(state_dict)
    else:
        optimizer = torch.optim.Adam(backbone.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        initial_params = spline.init_params(A)
        if 'xs' in initial_params:
            initial_params_ys = spline.init_params()['xs'].to(DEVICE)
        else:
            initial_params_ys = spline.init_params()['ys'].to(DEVICE)
        for i, (raw, enh) in enumerate(dataloader):
            raw = raw.to(DEVICE)
            params_tensor = backbone(raw)
            est_params = spline.get_params(params_tensor)
            loss = loss_fn(est_params['ys'], initial_params_ys)
            if 'xs' in initial_params:
                loss += loss_fn(est_params['xs'], initial_params_ys)
            if 'lambdas' in est_params:
                loss += loss_fn(est_params['lambdas'], torch.tensor(0.01).to(DEVICE))
            if 'A' in est_params and per_image:
                loss += loss_fn(est_params['A'], A)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("iter", i, loss.item())
            # break  # dev
            if loss < inittol:
                break

            torch.save(backbone.state_dict(), initckpt)

    # class IdentityBackbone(torch.nn.Module):
    #     def forward(self, x):
    #         return initial_params_ys.reshape(params_tensor.shape[1:])[None]
    
    # idbk = IdentityBackbone().to(DEVICE)

    print('validate weights...')
    # validate over one image
    val_img = Image.open(Path(DATASET_DIR) / "train" / "raw" / "004999.jpg")
    H, W = val_img.height, val_img.width
    val_img = val_img.resize((448,448))
    val_img = to_tensor(val_img).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)
    # validate_image(idbk, A, spline, val_img, Path('identity'))
    if not per_image:
        validate_image(backbone, A, spline, val_img, Path('initial'))
    else:
        perimage_validate_image(backbone, spline, val_img, Path('initial'))


    def loss_fn(rgb1, rgb2):
        return torch.norm(rgb2lab(rgb1) - rgb2lab(rgb2), dim=1).mean()
        # return squared_deltaE94(rgb2lab(rgb1), rgb2lab(rgb2)).mean()

    print('fit...')
    optimizer = torch.optim.Adam([{'params':backbone.parameters()}],
    lr=lr)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=lr,
    #     pct_start=0.05,
    #     steps_per_epoch=len(dataloader) // batch_size,
    #     epochs=n_epochs,
    # )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.1, patience=500, verbose=True
    )
    backbone = torch.compile(
        backbone, mode="reduce-overhead", disable=True
    )  # doesn't work, see https://github.com/pytorch/pytorch/issues/102539

    enhancer = fit(
        backbone,
        spline,
        dataloader,
        optimizer,
        scheduler,
        loss_fn,
        n_epochs=n_epochs,
        verbose=True,
        profiler=pr,
    )

    pr.disable()
    pr.dump_stats(f"profiler.prof")
    # breakpoint()
