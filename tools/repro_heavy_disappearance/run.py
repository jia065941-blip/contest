import argparse
from collections import deque
from contextlib import ExitStack
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def xyz(point):
    return tuple(getattr(point, axis)() for axis in 'xyz')


def native_snapshot(model, goal_ecf):
    from envengine.sdk.Util import UtilsPy

    state = model.getState()
    position, velocity = state.posEcf(), state.velEcf()
    lla = UtilsPy.CoordinateHelper.ecefToLla_py(position)
    nue = UtilsPy.CoordinateHelper.ecefToNueVelocity_py(velocity, lla.x(), lla.y())
    body = UtilsPy.CoordinateHelper.nueToBodyVelocity_py(nue, state.qbn())
    vx, vy, vz = xyz(body)
    return {'native_height_m': lla.z(), 'native_speed_mps': math.sqrt(vx*vx + vy*vy + vz*vz),
            'attack_angle_deg': math.atan2(vy, vx) * 57.3,
            'stage': state.stage(), 'remaining_native_distance_m': math.dist(xyz(position), xyz(goal_ecf))}


def run_native(case, output, correct_height):
    from envengine.sdk.Util import UtilsPy
    from envengine.simulator.models.ACMMissileModel import Missile

    name = 'native_height_control' if correct_height else 'native_original'
    start = list(case['native_initial_lla'])
    if correct_height:
        start[2] = case['desired_native_height_m']
    goal = case['native_target_lla']
    model = Missile()
    model.Init(case['native_step_seconds'], UtilsPy.Vector3D(*start), case['max_load'])
    model.SetDesiredHeight(case['desired_native_height_m'])
    model.SetDesiredSpeed(case['desired_native_speed_mps'])
    model.Save(False, case['heavy_id'])
    goal_vector = UtilsPy.Vector3D(*goal)
    model.Launch(goal_vector)
    goal_ecf = UtilsPy.CoordinateHelper.llaToEcef_py(goal_vector)
    tail = deque(maxlen=10)
    with (output / f'{name}.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=('time_s', 'return_value', 'native_height_m',
                               'native_speed_mps', 'attack_angle_deg', 'stage', 'remaining_native_distance_m'))
        writer.writeheader()
        for index in range(1, case['native_max_updates'] + 1):
            before = native_snapshot(model, goal_ecf)
            returned = model.Update()
            after = native_snapshot(model, goal_ecf)
            writer.writerow(dict(after, time_s=index * case['native_step_seconds'], return_value=returned))
            tail.append({'before': before, 'return_value': returned, 'after': after})
            if returned >= 0:
                break
    return {'case': name, 'initial_lla': start, 'end_seconds': index * case['native_step_seconds'],
            'return_value': returned, 'final': after, 'last_updates': list(tail),
            'angle_guard_observed': returned >= 0 and abs(before['attack_angle_deg']) > 80
                                    and math.isclose(returned, before['remaining_native_distance_m'], abs_tol=.00001)}


def run_engine(case, output):
    from envengine.environment.training_env import TrainingEnv
    from envengine.sdk.log import LogManager
    from envengine.sdk.Util import UtilsPy
    from envengine.sdk.writer import WriteConfig, init_writer
    from scenarios import load_scenario

    with ExitStack() as cleanup:
        cleanup.callback(LogManager(color_enabled=False, level=40).close)
        writer = init_writer(WriteConfig(enable_config=False, enable_state=False, enable_ai_action=False,
                                        enable_event=True, output_dir=str(output / 'engine_events'), verbose=False))
        cleanup.callback(writer.close)
        scene = load_scenario(HERE / 'scene.json')
        env = TrainingEnv(scene, record_logs=False)
        cleanup.callback(env.close)
        env.reset()
        manager = env.agent_manager
        manager.get_deployment_observation(seed=case['seed'])
        manager.apply_deployment({case['heavy_id']: tuple(case['game_initial_position_m'][:2])})
        simulator = env.get_simulator(case['heavy_id'])
        target = case['native_target_lla']
        goal_ecf = UtilsPy.CoordinateHelper.llaToEcef_py(UtilsPy.Vector3D(*target))
        rows = []
        before = env.get_observation()['entities'][case['heavy_id']]
        for step in range(1, case['engine_max_steps'] + 1):
            observation, _, done, _ = env.step()
            unit = observation['entities'][case['heavy_id']]
            rows.append(dict(native_snapshot(simulator.model, goal_ecf), step=step,
                             native_return=simulator.ret, health=unit['health'], visible=unit['isVisible'],
                             game_position=unit['pos']))
            if not unit['isVisible'] or unit['health'] <= 0:
                break
            if done:
                break
        write_json(output / 'engine_trace.json', rows)
        return {'initial_health': before['health'], 'disappearance_step': step if not unit['isVisible'] else None,
                'final': rows[-1], 'target_final_health': observation['entities'][case['target_id']]['health'],
                'red_strategy_count': len(env.agent_manager.get_registered_agents()), 'blue_policy_used': False}


def main():
    parser = argparse.ArgumentParser(description='复测E06真实部署点毁灭者提前结束，不修改引擎。')
    parser.add_argument('--mode', choices=('native', 'engine', 'all'), default='all')
    parser.add_argument('--output', type=Path, required=True, help='新建结果目录；已有目录不会覆盖')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        parser.error('原生模型要求Python 3.11.x')
    if sys.platform not in ('win32', 'linux'):
        parser.error('只支持项目提供的Windows和Linux原生模型')
    manifest = read_json(HERE / 'fixture_manifest.json')
    for name, expected in manifest['fixture_sha256'].items():
        if sha256(HERE / name) != expected:
            raise ValueError(f'固定复现输入发生变化：{name}')
    case = read_json(HERE / 'case.json')
    args.output.mkdir(parents=True, exist_ok=False)
    suffix = '_ACMMissilePy.pyd' if sys.platform == 'win32' else '_ACMMissilePy.so'
    model_path = 'envengine/simulator/models/ACMMissileModel/' + suffix
    runtime = {'model_sha256': sha256(ROOT / model_path),
               'coordinate_sha256': sha256(ROOT / 'envengine/simulator/util/scaled_coordinates.py'),
               'python': sys.version, 'platform': platform.platform(),
               'fixture_sha256': manifest['fixture_sha256']}
    runtime['model_matches_reference'] = runtime['model_sha256'] == manifest['reference_runtime_sha256'][model_path]
    runtime['coordinate_matches_reference'] = runtime['coordinate_sha256'] == manifest['reference_coordinate_sha256']
    result = {'case_id': case['case_id'], 'runtime': runtime, 'mode': args.mode,
              'engine_modified': False, 'native_uses_frozen_inputs': True}
    if args.mode in ('native', 'all'):
        result['native_original'] = run_native(case, args.output, False)
        result['native_height_control'] = run_native(case, args.output, True)
    if args.mode in ('engine', 'all'):
        result['engine'] = run_engine(case, args.output)
    write_json(args.output / 'summary.json', result)
    if 'native_original' in result:
        print(f"原生模型：{result['native_original']['end_seconds']:.2f}秒结束，迎角退出={result['native_original']['angle_guard_observed']}")
        print(f"高度对照：{result['native_height_control']['end_seconds']:.2f}秒结束")
    if 'engine' in result:
        print(f"真实引擎：消失步号={result['engine']['disappearance_step']}，血量={result['engine']['final']['health']}")
    print(f'结果：{args.output.resolve() / "summary.json"}')


if __name__ == '__main__':
    main()
