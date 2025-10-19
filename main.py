# -*- coding: utf-8 -*-
import argparse
import configparser
import enum
import sys
import threading
import time
import traceback
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass
from queue import PriorityQueue, Empty
from threading import RLock
from typing import Any, Union

from tqdm import tqdm

from api.answer import Tiku
from api.base import Chaoxing, Account, StudyResult
from api.exceptions import LoginError, InputFormatError
from api.logger import logger
from api.notification import Notification

# 定义一个哨兵值（sentinel value），用于通知线程可以退出了
_SHUTDOWN_SENTINEL = None


class ChapterResult(enum.Enum):
    SUCCESS = 0,
    ERROR = 1,
    NOT_OPEN = 2,
    PENDING = 3


def log_error(func):
    """一个装饰器，用于捕获并记录线程中发生的异常"""

    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except BaseException as e:
            logger.error(f"Error in thread {threading.current_thread().name}: {e}")
            traceback.print_exception(type(e), e, e.__traceback__)
            # 在实际应用中，你可能还想在这里处理线程的异常退出
            pass

    return wrapper


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Samueli924/chaoxing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--use-cookies", action="store_true", help="使用cookies登录")
    parser.add_argument("-c", "--config", type=str, default=None, help="使用配置文件运行程序")
    parser.add_argument("-u", "--username", type=str, default=None, help="手机号账号")
    parser.add_argument("-p", "--password", type=str, default=None, help="登录密码")
    parser.add_argument("-l", "--list", type=str, default=None, help="要学习的课程ID列表, 以 , 分隔")
    parser.add_argument("-s", "--speed", type=float, default=1.0, help="视频播放倍速 (默认1, 最大2)")
    parser.add_argument("-j", "--jobs", type=int, default=4, help="同时进行的章节任务数 (默认4)")
    parser.add_argument("-v", "--verbose", "--debug", action="store_true", help="启用调试模式, 输出DEBUG级别日志")
    parser.add_argument(
        "-a", "--notopen-action", type=str, default="retry",
        choices=["retry", "ask", "continue"],
        help="遇到关闭任务点时的行为: retry-重试, ask-询问, continue-继续"
    )
    if len(sys.argv) == 2 and sys.argv[1] in {"-h", "--help"}:
        parser.print_help()
        sys.exit(0)
    return parser.parse_args()


def load_config_from_file(config_path):
    """从配置文件加载设置"""
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf8")
    common_config: dict[str, Any] = {}
    tiku_config: dict[str, Any] = {}
    notification_config: dict[str, Any] = {}
    if config.has_section("common"):
        common_config = dict(config.items("common"))
        if "course_list" in common_config and common_config["course_list"]:
            common_config["course_list"] = [item.strip() for item in
                                            common_config["course_list"].split(",") if item.strip()]
        if "speed" in common_config:
            common_config["speed"] = float(common_config["speed"])
        if "jobs" in common_config:
            common_config["jobs"] = int(common_config["jobs"])
        if "notopen_action" not in common_config:
            common_config["notopen_action"] = "retry"
        if "use_cookies" in common_config:
            common_config["use_cookies"] = str_to_bool(common_config["use_cookies"])
        if "username" in common_config and common_config["username"] is not None:
            common_config["username"] = common_config["username"].strip()
        if "password" in common_config and common_config["password"] is not None:
            common_config["password"] = common_config["password"].strip()
    if config.has_section("tiku"):
        tiku_config = dict(config.items("tiku"))
        for key in ["delay", "cover_rate"]:
            if key in tiku_config:
                tiku_config[key] = float(tiku_config[key])
    if config.has_section("notification"):
        notification_config = dict(config.items("notification"))
    return common_config, tiku_config, notification_config


def build_config_from_args(args):
    """从命令行参数构建配置"""
    common_config = {
        "use_cookies": args.use_cookies,
        "username": args.username,
        "password": args.password,
        "course_list": [item.strip() for item in args.list.split(",") if item.strip()] if args.list else None,
        "speed": args.speed if args.speed else 1.0,
        "jobs": args.jobs,
        "notopen_action": args.notopen_action if args.notopen_action else "retry"
    }
    return common_config, {}, {}


def init_config():
    """初始化配置"""
    args = parse_args()
    if args.config:
        return load_config_from_file(args.config)
    else:
        return build_config_from_args(args)


def init_chaoxing(common_config, tiku_config):
    """初始化超星实例"""
    username = common_config.get("username", "")
    password = common_config.get("password", "")
    use_cookies = common_config.get("use_cookies", False)
    if (not username or not password) and not use_cookies:
        username = input("请输入你的手机号, 按回车确认\n手机号:")
        password = input("请输入你的密码, 按回车确认\n密码:")
    account = Account(username, password)
    tiku = Tiku()
    tiku.config_set(tiku_config)
    tiku = tiku.get_tiku_from_config()
    tiku.init_tiku()
    query_delay = tiku_config.get("delay", 0)
    chaoxing = Chaoxing(account=account, tiku=tiku, query_delay=query_delay)
    return chaoxing


def process_job(chaoxing: Chaoxing, course: dict, job: dict, job_info: dict, speed: float) -> StudyResult:
    """处理单个任务点（视频、文档等）"""
    job_type = job.get("type")
    title = course.get('title', '未知章节')
    job_id = job.get('jobid', '未知ID')

    if job_type == "video":
        logger.trace(f"识别到视频任务, 任务章节: {title} 任务ID: {job_id}")
        video_result = chaoxing.study_video(course, job, job_info, _speed=speed, _type="Video")
        if video_result.is_failure():
            logger.warning(f"视频任务解码失败, 尝试音频模式: {title} 任务ID: {job_id}")
            video_result = chaoxing.study_video(course, job, job_info, _speed=speed, _type="Audio")
        if video_result.is_failure():
            logger.warning(f"异常任务 -> 任务章节: {title} 任务ID: {job_id}, 已跳过")
        return video_result
    elif job_type == "document":
        logger.trace(f"识别到文档任务, 任务章节: {title} 任务ID: {job_id}")
        return chaoxing.study_document(course, job)
    elif job_type == "workid":
        logger.trace(f"识别到章节测验任务, 任务章节: {title}")
        return chaoxing.study_work(course, job, job_info)
    elif job_type == "read":
        logger.trace(f"识别到阅读任务, 任务章节: {title}")
        return chaoxing.study_read(course, job, job_info)
    else:
        logger.error(f"未知任务类型: {job_type}, 任务章节: {title}, 已跳过")
        return StudyResult.ERROR


def process_chapter(chaoxing: Chaoxing, course: dict, point: dict, speed: float) -> ChapterResult:
    """处理单个章节的所有任务点"""
    chapter_title = point.get("title", "未知章节")
    logger.info(f'开始处理章节: {chapter_title}')

    if point.get("has_finished"):
        logger.info(f'章节: {chapter_title} 已完成所有任务点')
        return ChapterResult.SUCCESS

    chaoxing.rate_limiter.limit_rate(random_time=True, random_min=0, random_max=0.2)

    try:
        jobs, job_info = chaoxing.get_job_list(course, point)
    except Exception as e:
        logger.error(f"获取章节任务列表失败: {chapter_title}, 错误: {e}")
        return ChapterResult.ERROR

    if job_info and job_info.get("notOpen", False):
        logger.warning(f"章节: {chapter_title} 未开放")
        return ChapterResult.NOT_OPEN

    if not jobs:
        logger.info(f"章节: {chapter_title} 没有需要完成的任务点")
        return ChapterResult.SUCCESS

    job_results: list[StudyResult] = []
    # 使用线程池并行处理一个章节内的所有任务点
    with ThreadPoolExecutor(max_workers=5, thread_name_prefix=f"Job-{chapter_title[:10]}") as executor:
        futures = {executor.submit(process_job, chaoxing, course, job, job_info, speed): job for job in jobs}
        for future in futures:
            try:
                result = future.result()
                job_results.append(result)
            except Exception as e:
                logger.error(f"任务点执行异常: {e}")
                job_results.append(StudyResult.ERROR)

    # 如果任何一个任务点失败了，则认为整个章节处理失败
    if any(result.is_failure() for result in job_results):
        return ChapterResult.ERROR

    return ChapterResult.SUCCESS


@dataclass(order=True)
class ChapterTask:
    """用于在优先队列中排序的章节任务对象"""
    index: int
    point: Any = dataclasses.field(compare=False)
    result: ChapterResult = ChapterResult.PENDING
    tries: int = 0


class JobProcessor:
    """多线程处理课程所有章节的核心类"""
    def __init__(self, chaoxing: Chaoxing, course: dict, tasks: list[ChapterTask], config: dict):
        self.chaoxing = chaoxing
        self.course = course
        self.speed = config["speed"]
        self.max_tries = 5
        self.failed_tasks: list[ChapterTask] = []
        self.task_queue: PriorityQueue[Union[ChapterTask, None]] = PriorityQueue()
        self.retry_queue: PriorityQueue[Union[ChapterTask, None]] = PriorityQueue()
        self.threads: list[threading.Thread] = []
        self.worker_num = config["jobs"]
        self.config = config
        self._shutdown_event = threading.Event()

    def run(self):
        """启动所有线程并开始处理任务"""
        # 启动重试线程和工作线程
        retry_thread = threading.Thread(target=self.retry_thread, daemon=True, name="RetryThread")
        retry_thread.start()

        for i in range(self.worker_num):
            thread = threading.Thread(target=self.worker_thread, daemon=True, name=f"Worker-{i}")
            self.threads.append(thread)
            thread.start()

        # 将所有初始任务放入队列
        for task in tasks:
            self.task_queue.put(task)

        # 等待所有任务被处理（task_done被调用）
        self.task_queue.join()

        # 所有任务处理完毕，通知所有线程关闭
        logger.info("所有章节任务已完成，正在关闭线程...")
        self._shutdown_event.set()
        self.retry_queue.put(_SHUTDOWN_SENTINEL) # 放入哨兵值以唤醒可能阻塞的重试线程

        # 等待所有工作线程和重试线程结束
        for thread in self.threads:
            thread.join()
        retry_thread.join()
        logger.info("所有线程已安全关闭。")

    @log_error
    def worker_thread(self):
        """工作线程，从主队列获取并处理章节任务"""
        while not self._shutdown_event.is_set():
            try:
                task = self.task_queue.get(timeout=1)
            except Empty:
                continue # 队列空，继续循环检查是否需要关闭

            try:
                task.result = process_chapter(self.chaoxing, self.course, task.point, self.speed)

                if task.result == ChapterResult.SUCCESS:
                    logger.debug(f"Task success: {task.point['title']}")
                elif task.result == ChapterResult.NOT_OPEN:
                    if self.config["notopen_action"] == "continue":
                        logger.warning(f"章节未开放: {task.point['title']}, 已根据配置跳过。")
                    else:
                        task.tries += 1
                        if task.tries >= self.max_tries:
                             logger.error(f"章节 {task.point['title']} 重试多次后仍未开放，已放弃。")
                        else:
                            self.retry_queue.put(task) # 放入重试队列
                elif task.result == ChapterResult.ERROR:
                    task.tries += 1
                    logger.warning(f"任务失败，正在重试: {task.point['title']} ({task.tries}/{self.max_tries})")
                    if task.tries >= self.max_tries:
                        logger.error(f"任务达到最大重试次数: {task.point['title']}")
                        self.failed_tasks.append(task)
                    else:
                        self.retry_queue.put(task) # 放入重试队列
            finally:
                self.task_queue.task_done()

    @log_error
    def retry_thread(self):
        """重试线程，将失败的任务在延迟后重新放回主队列"""
        while not self._shutdown_event.is_set():
            task = self.retry_queue.get()
            if task is _SHUTDOWN_SENTINEL:
                break
            
            logger.info(f"章节 {task.point['title']} 将在10秒后重试...")
            time.sleep(10) # 重试前等待
            self.task_queue.put(task)


def process_course(chaoxing: Chaoxing, course: dict, config: dict):
    """处理单门课程的所有章节"""
    logger.info(f"开始学习课程: {course['title']}")

    try:
        point_list = chaoxing.get_course_point(course["courseId"], course["clazzId"], course["cpi"])
    except Exception as e:
        logger.error(f"无法获取课程《{course['title']}》的章节列表: {e}")
        return

    tasks = [ChapterTask(point=point, index=i) for i, point in enumerate(point_list.get("points", []))]
    if not tasks:
        logger.info(f"课程《{course['title']}》没有可学习的章节。")
        return

    # 使用 tqdm 创建进度条
    with tqdm(total=len(tasks), desc=f"学习课程: {course['title']}") as pbar:
        # 创建一个回调函数来更新进度条
        original_task_done = PriorityQueue.task_done
        def new_task_done(self):
            if self.qsize() < pbar.total: # 只有当任务被真正消耗时才更新
                 pbar.update(1)
            original_task_done(self)
        
        # 猴子补丁：临时替换 task_done 方法以更新进度条
        PriorityQueue.task_done = new_task_done
        
        processor = JobProcessor(chaoxing, course, tasks, config)
        processor.run()

        # 恢复原始方法
        PriorityQueue.task_done = original_task_done


def filter_courses(all_course, course_list):
    """根据提供的列表筛选课程"""
    if not course_list:
        print("*" * 10 + " 课程列表 " + "*" * 10)
        for course in all_course:
            print(f"ID: {course['courseId']} 课程名: {course['title']}")
        print("*" * 28)
        try:
            course_ids_str = input("请输入想学习的课程ID,用逗号隔开 (直接回车则学习所有课程):\n")
            if not course_ids_str.strip():
                return all_course
            course_list = [item.strip() for item in course_ids_str.split(",")]
        except Exception as e:
            raise InputFormatError(f"输入格式错误: {e}")

    course_task = []
    selected_ids = set()
    for course in all_course:
        if course["courseId"] in course_list and course["courseId"] not in selected_ids:
            course_task.append(course)
            selected_ids.add(course["courseId"])
    
    if not course_task:
        logger.warning("根据你输入的ID，没有找到任何匹配的课程。将学习所有课程。")
        return all_course
        
    return course_task


def format_time(num, *args, **kwargs):
    """格式化tqdm的时间显示"""
    total_time = round(num)
    sec = total_time % 60
    mins = (total_time % 3600) // 60
    hrs = total_time // 3600
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{sec:02d}"
    return f"{mins:02d}:{sec:02d}"


def main():
    """主程序入口"""
    try:
        common_config, tiku_config, notification_config = init_config()
        common_config["speed"] = min(2.0, max(1.0, common_config.get("speed", 1.0)))
        common_config["notopen_action"] = common_config.get("notopen_action", "retry")
        common_config["jobs"] = common_config.get("jobs", 4)
        
        chaoxing = init_chaoxing(common_config, tiku_config)
        
        notification = Notification()
        notification.config_set(notification_config)
        notification = notification.get_notification_from_config()
        notification.init_notification()
        
        _login_state = chaoxing.login(login_with_cookies=common_config.get("use_cookies", False))
        if not _login_state["status"]:
            raise LoginError(_login_state["msg"])
        
        all_course = chaoxing.get_course_list()
        course_task = filter_courses(all_course, common_config.get("course_list"))
        
        logger.info(f"课程列表过滤完毕, 当前课程任务数量: {len(course_task)}")

        original_sizeof = tqdm.format_sizeof
        tqdm.format_sizeof = format_time
        
        for course in course_task:
            process_course(chaoxing, course, common_config)
        
        tqdm.format_sizeof = original_sizeof

        logger.info("所有课程学习任务已完成")
        notification.send("chaoxing : 所有课程学习任务已完成")
        
    except (SystemExit, KeyboardInterrupt) as e:
        logger.warning(f"程序被用户中断。")
        sys.exit(0)
    except BaseException as e:
        logger.error(f"发生未处理的异常: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        try:
            notification.send(f"chaoxing : 出现错误 {type(e).__name__}: {e}\n{traceback.format_exc()}")
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
