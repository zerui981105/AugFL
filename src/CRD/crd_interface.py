from CRD.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from CRD.cifar10 import get_cifar10_dataloaders, get_cifar10_dataloaders_sample
from CRD.fmnist import get_fmnist_dataloaders, get_fmnist_dataloaders_sample
from CRD.emnist import get_emnist_dataloaders, get_emnist_dataloaders_sample
import torch
from torch import nn
from CRD.criterion import CRDLoss
import torch.optim as optim
import torch.backends.cudnn as cudnn
from CRD.loops import train_distill as train,validate
from CRD.util import adjust_learning_rate
from CRD.KD import DistillKL
import time
from CRD.models import model_dict
def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict['resnet32x4'](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model

class CRD(object):
    def __init__(self, opt, model_s):
        # print(model_s.state_dict()['conv1.weight'][0])
        self.best_acc = 0
        self.opt = opt
        n_cls = 10
        if opt.dataset == 'cifar100':
            if opt.distill in ['crd']:
                self.train_loader, self.val_loader, self.n_data = get_cifar100_dataloaders_sample(batch_size=128,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
            else:
                self.train_loader, self.val_loader, self.n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
            n_cls = 100
        if opt.dataset == 'fmnist':
            if opt.distill in ['crd']:
                self.train_loader, self.val_loader, self.n_data = get_fmnist_dataloaders_sample(batch_size=128,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
            else:
                self.train_loader, self.val_loader, self.n_data = get_fmnist_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
            n_cls = 10
        if opt.dataset == 'emnist':
            if opt.distill in ['crd']:
                self.train_loader, self.val_loader, self.n_data = get_emnist_dataloaders_sample(batch_size=128,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
            else:
                self.train_loader, self.val_loader, self.n_data = get_emnist_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
            n_cls = 26
        # else:
        #     raise NotImplementedError(opt.dataset)
        self.model_t = load_teacher(opt.path_t, n_cls)
        model_s = model_s
        data = torch.randn(2, 3, 32, 32)
        self.model_t.eval()
        model_s.eval()
        feat_t, _ = self.model_t(data, is_feat=True)

        feat_s, _ = model_s(data, is_feat=True)
        self.opt.s_dim = feat_s[-1].shape[1]
        self.opt.t_dim = feat_t[-1].shape[1]
        self.opt.n_data = self.n_data
        self.criterion_kd = CRDLoss(opt)
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_KL = DistillKL(opt.kd_T)
        if torch.cuda.is_available():
            self.model_t.cuda()
            self.criterion_cls.cuda()
            cudnn.benchmark = True
        teacher_acc, _ = validate(self.val_loader, self.model_t, self.criterion_cls, self.opt)
        print('teacher accuracy: ', teacher_acc)

    def train(self, model_s):
        # print('%%%%%%%%%%%%%%%%')
        # print(model_s.state_dict()['conv1.weight'][0])
        # print('%%%%%%%%%%%%%%%%')
        module_list = nn.ModuleList([])
        module_list.append(model_s)
        trainable_list = nn.ModuleList([])
        trainable_list.append(model_s)
        module_list.append(self.criterion_kd.embed_s)
        module_list.append(self.criterion_kd.embed_t)
        trainable_list.append(self.criterion_kd.embed_s)
        trainable_list.append(self.criterion_kd.embed_t)
        criterion_list = nn.ModuleList([])
        criterion_list.append(self.criterion_cls)  # classification loss
        criterion_list.append(self.criterion_KL)
        criterion_list.append(self.criterion_kd)
        optimizer = optim.SGD(trainable_list.parameters(),
                              lr=self.opt.learning_rate,
                              momentum=self.opt.momentum,
                              weight_decay=self.opt.weight_decay)
        module_list.append(self.model_t)
        if torch.cuda.is_available():
            module_list.cuda()
            criterion_list.cuda()
            cudnn.benchmark = True

        for epoch in range(1, self.opt.epochs + 1):

            adjust_learning_rate(epoch, self.opt, optimizer)
            print("==> training...")

            time1 = time.time()
            train_acc, train_loss, grad_model_s = train(epoch, self.train_loader, module_list, criterion_list, optimizer, self.opt,
                                          trainable_list)
            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            # logger.log_value('train_acc', train_acc, epoch)
            # logger.log_value('train_loss', train_loss, epoch)

            test_acc, test_loss = validate(self.val_loader, model_s, self.criterion_cls, self.opt)

            # logger.log_value('test_acc', test_acc, epoch)
            # logger.log_value('test_loss', test_loss, epoch)
            # logger.log_value('test_acc_top5', tect_acc_top5, epoch)

            # save the best model
            if test_acc > self.best_acc:
                self.best_acc = test_acc
            #     state = {
            #         'epoch': epoch,
            #         'model': model_s.state_dict(),
            #         'best_acc': best_acc,
            #     }
            #     save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            #     print('saving the best model!')
            #     torch.save(state, save_file)
            #
            # # regular saving
            # if epoch % opt.save_freq == 0:
            #     print('==> Saving...')
            #     state = {
            #         'epoch': epoch,
            #         'model': model_s.state_dict(),
            #         'accuracy': test_acc,
            #     }
            #     save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            #     torch.save(state, save_file)

        # This best accuracy is only for printing purpose.
        # The results reported in the paper/README is from the last epoch.
            print('best accuracy:', self.best_acc)
        return grad_model_s





def run_CRD(opt, model_s):
    best_acc = 0
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    model_t = load_teacher(opt.path_t, n_cls)
    # model_s = model_dict[opt.model_s](num_classes=n_cls)
    model_s = model_s

    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)

    feat_s, _ = model_s(data, is_feat=True)


    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    # criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    # criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    teacher_acc, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)

    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt,trainable_list)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # logger.log_value('train_acc', train_acc, epoch)
        # logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        # logger.log_value('test_acc', test_acc, epoch)
        # logger.log_value('test_loss', test_loss, epoch)
        # logger.log_value('test_acc_top5', tect_acc_top5, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
        #     state = {
        #         'epoch': epoch,
        #         'model': model_s.state_dict(),
        #         'best_acc': best_acc,
        #     }
        #     save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
        #     print('saving the best model!')
        #     torch.save(state, save_file)
        #
        # # regular saving
        # if epoch % opt.save_freq == 0:
        #     print('==> Saving...')
        #     state = {
        #         'epoch': epoch,
        #         'model': model_s.state_dict(),
        #         'accuracy': test_acc,
        #     }
        #     save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        #     torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)

    # save model
    # state = {
    #     'opt': opt,
    #     'model': model_s.state_dict(),
    # }
    # save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    # torch.save(state, save_file)
def run_CRD_all(opt, model_s):
    best_acc = 0
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample_all(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_all(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    model_t = load_teacher(opt.path_t, n_cls)
    # model_s = model_dict[opt.model_s](num_classes=n_cls)
    model_s = model_s

    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)

    feat_s, _ = model_s(data, is_feat=True)


    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    # criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    # criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    teacher_acc, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)

    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt,trainable_list)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # logger.log_value('train_acc', train_acc, epoch)
        # logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        # logger.log_value('test_acc', test_acc, epoch)
        # logger.log_value('test_loss', test_loss, epoch)
        # logger.log_value('test_acc_top5', tect_acc_top5, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
        #     state = {
        #         'epoch': epoch,
        #         'model': model_s.state_dict(),
        #         'best_acc': best_acc,
        #     }
        #     save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
        #     print('saving the best model!')
        #     torch.save(state, save_file)
        #
        # # regular saving
        # if epoch % opt.save_freq == 0:
        #     print('==> Saving...')
        #     state = {
        #         'epoch': epoch,
        #         'model': model_s.state_dict(),
        #         'accuracy': test_acc,
        #     }
        #     save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        #     torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)