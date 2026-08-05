import logging
import requests
from envengine import sdk, Profile
import json
import os
from urllib.parse import urlparse
from envengine.engine import Engine

logging.basicConfig(level=logging.INFO)


def read_profile(url: str) -> Profile:
    """
    通过本地文件url读取或者通过远程url读取
    :param url:
    :return:
    """
    # 判断是否为有效的URL（包含协议）
    parsed_url = urlparse(url)

    if parsed_url.scheme in ['http', 'https']:  # 远程文件路径
        response = requests.get(url)
        if response.status_code == 200:
            profile_data = response.json()['data']
            return Profile.from_dict(profile_data)
        else:
            raise Exception(f"无法从远程URL获取数据: {response.status_code}")
    elif os.path.exists(url):  # 本地文件路径
        with open(url, 'r', encoding='utf-8') as file:
            profile_data = json.load(file)
            return Profile.from_dict(profile_data)
    else:
        raise ValueError(f"无效的URL或文件路径: {url}")


if __name__ == '__main__':
    profile: Profile = read_profile("./scenarios/platform.json")

    # profile: sdk.base_struct.profile.profile = read_profile(
    #     "http://192.168.1.16:30510/experiment/project/result/get?platformId=815&experimentId=42&imageName=platform")
    print(profile)

    # 实例化仿真引擎
    engine: Engine = Engine(profile)
    engine.run()
