from lib.config import args, cfg
from lib import make_network, make_data_loader, make_evaluator
from lib.mesh_utils import extract_mesh, refuse, transform
from lib.net_utils import load_network
import open3d as o3d


def run():
    network = make_network(cfg).cuda()
    load_network(
        network,
        cfg.trained_model_dir,
        resume=cfg.resume,
        epoch=cfg.test.epoch
    )
    network.eval()
    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)

    mesh = extract_mesh(network.model.sdf_net)
    mesh = refuse(mesh, data_loader)
    mesh = transform(mesh, cfg.test_dataset.scale, cfg.test_dataset.offset)

    if args.output_mesh != '':
        o3d.io.write_triangle_mesh(args.output_mesh, mesh)

    mesh_gt = o3d.io.read_triangle_mesh(f'{cfg.test_dataset.data_root}/{cfg.test_dataset.scene}/gt.obj')
    evaluate_result = evaluator.evaluate(mesh, mesh_gt)
    for k, v in evaluate_result.items():
        print(f'{k:7s}: {v:1.3f}')

        
if __name__ == '__main__':
    run()
