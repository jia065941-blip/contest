import msgpack


def read_msgpackl(filepath):
    with open(filepath, 'rb') as f:
        unpacker = msgpack.Unpacker(f, strict_map_key=False)
        for item in unpacker:  # 遍历所有数据
            print(item)




# 使用
read_msgpackl("G:\\PycharmProjects\\competition-platform-env\\results\\20260705194827\\1.json")

