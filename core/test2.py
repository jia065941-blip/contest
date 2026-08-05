import random

import fastdisjointset


def can_communicate(pos1, pos2, distance_limit):
    """根据实际位置判断能否通信"""
    dist = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    return dist <= distance_limit


def demo_by_position():
    print("=" * 50)
    print("基于二维位置分组（真实距离限制）")
    print("=" * 50)

    # 生成100个飞行器的位置（分成5个簇）
    positions = {}
    cluster_centers = [(0, 0), (100, 0), (0, 100), (100, 100), (50, 50)]

    flyer_id = 0
    for center_x, center_y in cluster_centers:
        for _ in range(20):  # 每个簇20个飞行器
            # 在簇中心周围随机分布
            x = center_x + random.uniform(-20, 20)
            y = center_y + random.uniform(-20, 20)
            positions[flyer_id] = (x, y)
            flyer_id += 1

    ds = fastdisjointset.DisjointSet(100)
    distance_limit = 30  # 通信距离30

    for i in range(100):
        for j in range(i + 1, 100):
            if can_communicate(positions[i], positions[j], distance_limit):
                ds.union(i, j)

    groups = ds.sets()
    print(f"通信距离限制: {distance_limit}")
    print(f"群的数量：{len(groups)}")
    for idx, group in enumerate(groups):
        print(f"  群{idx + 1}: {sorted(group)} (共{len(group)}个飞行器)")
    print()


if __name__ == "__main__":
    demo_by_position()
