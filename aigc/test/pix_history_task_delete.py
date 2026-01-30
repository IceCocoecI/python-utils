import requests
import time
from urllib.parse import quote

# 配置信息
AUTH_TOKEN = "user:1306-hoIj3Rb7MOeatCwMx3Srw"
BASE_URL = "https://api.useapi.net/v2/pixverse/scheduler"


def fetch_tasks_and_emails():
    """
    获取正在执行的任务列表和可用资源的邮箱集合
    返回 (executing_tasks, available_emails)
    """
    url = f"{BASE_URL}/available"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AUTH_TOKEN}"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        executing_tasks = data.get("executing", [])
        available_emails = {item["email"] for item in data.get("available", [])}
        return executing_tasks, available_emails
    except requests.exceptions.RequestException as e:
        print(f"获取任务列表失败: {e}")
        return [], set()


def delete_task(video_id):
    """
    删除单个任务
    """
    # 移除URL编码，直接使用原始video_id构建URL
    url = f"{BASE_URL}/{video_id}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AUTH_TOKEN}"
    }

    try:
        response = requests.delete(url, headers=headers)
        response.raise_for_status()
        print(f"成功删除任务: {video_id}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"删除任务失败 {video_id}: {e}")
        return False


def main():
    # 获取任务列表和可用邮箱
    executing_tasks, available_emails = fetch_tasks_and_emails()

    # 过滤需要删除的任务
    filtered_tasks = []
    for task in executing_tasks:
        video_id = task.get("video_id")
        if not video_id:
            continue

        try:
            # 提取video_id中的邮箱部分
            email_part = video_id.split("pixverse:")[1].split("-video:")[0]

            # 如果提取的邮箱在可用邮箱集合中，跳过该任务
            if email_part in available_emails:
                print(f"跳过任务: {video_id} (关联邮箱 {email_part} 正在执行)")
                continue

        except IndexError:
            print(f"无法解析video_id: {video_id}")
            continue

        filtered_tasks.append(task)

    # 执行删除操作
    for task in filtered_tasks:
        video_id = task.get("video_id")
        delete_task(video_id)
        time.sleep(0.05)  # 防止API限流


if __name__ == "__main__":
    main()
