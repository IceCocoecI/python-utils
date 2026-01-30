########################  降级测试代码  ######################

import logging
import os
import random
import uuid
import asyncio
import cv2

# 假设外部依赖已经导入
# from your_project import ... (SubFuncType, ApiPlatformRedisKey, etc.)

# ==========================================
# 1. 配置与映射 (Configuration & Mappings)
# ==========================================

# 降级配置表：定义服务失败后的跳转路径
# 结构: KEY -> List[{"target": 下一个服务名, "alert": 是否告警}]
FALLBACK_CONFIG = {
    # NanoBanana AI GptStore -> FalAI (告警) -> Replicate (告警)
    "NANO_BANANA_AI_GPTSTORE": [
        {"target": "NANO_BANANA_FAL_AI", "alert": True},
        {"target": "NANO_BANANA_REPLICATE", "alert": True}
    ],
    # Gemini -> Replicate (告警)
    "GEMINI_AI_GPTSTORE": [
        {"target": "NANO_BANANA_REPLICATE", "alert": True}
    ],
    # 默认无降级
    "DEFAULT": []
}

# 需要预先上传到国内 OSS 的 sub_func 集合
OSS_UPLOAD_REQUIRED_SUBFUNCS = {
    "SEEDEDIT_OFFICIAL",
    "SEEDEDIT_AI_GPTSTORE",
    "FLUX_KONTEXT_AI_GPTSTORE",
    "LOCAL_QWEN_EDIT"
}

# 错误匹配模式 (提取至全局或静态配置)
ERROR_PATTERNS = [
    "Error forwarded from LMRoot",
    "当前分组上游负载已饱和",
    "Nano Banana Aigptstore POST Request. TimeoutError",
    "Nano Banana Aigptstore POST Request: RequestError",
    "No valid image URL or base64 data found",
]
# 包含匹配模式 (需配合 helper 函数)
CONTAINS_PATTERNS = [
    ("HTTPStatusError", "500"),
    ("HTTPStatusError", "502"),
    ("HTTPStatusError", "503"),
    ("HTTPStatusError", "504"),
]


# ==========================================
# 2. 上下文管理 (Context)
# ==========================================

class ExecutionContext:
    """
    执行上下文，负责解析输入参数、处理随机逻辑、持有中间状态
    """

    def __init__(self, input_dict, logging_obj):
        self.input_dict = input_dict
        self.logging = logging_obj
        self.work_id = input_dict.get('work_id')
        self.sub_func = input_dict.get('subFunc')
        self.params = input_dict.get("parameter_dict", {})
        self.dynamic_params = input_dict.get('dynamicParams', {})
        self.input_path = input_dict.get("input_path")

        # --- 初始化处理逻辑 ---
        self._init_prompt()
        self._init_reference_images()

        # --- 中间状态容器 ---
        self.preprocessed_user_image_paths = []  # 预处理后的图片
        self.oss_image_url_list = []  # 临时OSS链接
        self.upload_img_path_list = []  # 最终下载/生成的本地路径
        self.final_result = {}  # API 原始返回

    def _init_prompt(self):
        """处理随机提示词逻辑"""
        self.prompt = self.params.get("prompt", "")
        multiple_random_prompt = self.params.get("multiple_random_prompt", '[]')
        multiple_random_prompt = eval(multiple_random_prompt) if isinstance(multiple_random_prompt,
                                                                            str) else multiple_random_prompt
        prompt_sequential = self.params.get("prompt_sequential_execution", False)

        if len(multiple_random_prompt) > 0 and not prompt_sequential:
            self.prompt = random.choice(multiple_random_prompt)
            self.logging.info(f"{self.work_id} multiple_random_prompt choice: {self.prompt}")
            self.params["prompt"] = self.prompt  # 更新回 params 以防 handler 使用

    def _init_reference_images(self):
        """处理参考图分割与随机选择逻辑"""
        ref_imgs = self.params.get('reference_images', [])
        # 假设 split_string 是外部工具函数
        if isinstance(ref_imgs, str):
            ref_imgs = split_string(ref_imgs, ",")

        if len(ref_imgs) > 1:
            # 随机选择一张
            ref_imgs = [random.choice(ref_imgs)]
            self.logging.info(f"reference_image_paths choice: {ref_imgs}")

        self.reference_image_paths = ref_imgs


# ==========================================
# 3. 处理器基类 (Base Handler)
# ==========================================

class BaseImageEditHandler:
    def __init__(self, context: ExecutionContext):
        self.ctx = context

    def execute(self) -> dict:
        raise NotImplementedError

    def should_fallback(self, result: dict) -> tuple[bool, str]:
        """判断是否需要触发降级"""
        # 1. 基础 Flag 判断
        if result.get('flag') != 1:
            return True, result.get('error_message', 'Unknown Error')

        # 2. 模式匹配 (Pattern Matching)
        # 假设 create_exact_match_pattern 等是外部函数，这里模拟逻辑
        # 实际使用时，建议将 patterns 传入 check_pattern_match
        # 这里简化处理，复用原来的逻辑思路

        # 构造原代码中的 patterns 列表
        patterns = [create_exact_match_pattern(p) for p in ERROR_PATTERNS]
        patterns.extend([create_contains_all_pattern(p[0], p[1]) for p in CONTAINS_PATTERNS])

        if check_pattern_match(patterns, result):
            return True, result.get('error_message', 'Pattern Matched Error')

        return False, None


# ==========================================
# 4. 具体业务 Handler 实现
# ==========================================

class SeedEditOfficialHandler(BaseImageEditHandler):
    def execute(self):
        platform = ApiPlatformRedisKey.BYTEDANCE_SEEDEDIT
        combined_key = APIPlatformAccountManager.get_api_platform_token(platform, self.ctx.work_id)
        access_key, secret_key = combined_key.split("#")
        seed_edit_api = SeedEditAPI(access_key=access_key, secret_key=secret_key)
        return asyncio.run(seed_edit_api.task_pipeline(
            image_url=self.ctx.oss_image_url_list[0],
            prompt=self.ctx.prompt,
            negative_prompt=self.ctx.params.get("negative_prompt", ""),
            work_id=self.ctx.work_id
        ))


class SeedEditGptStoreHandler(BaseImageEditHandler):
    def execute(self):
        sub_func = self.ctx.sub_func
        flux_model = self.ctx.params.get("flux_model", "flux-kontext-pro")

        if sub_func == SubFuncType.SEEDEDIT_AI_GPTSTORE.value:
            custom_type = SubFuncType.SEEDEDIT_AI_GPTSTORE
            model = "seededit"
        else:
            custom_type = SubFuncType.FLUX_KONTEXT_AI_GPTSTORE
            model = flux_model

        platform_token = APIPlatformAccountManager.get_api_platform_token(ApiPlatformRedisKey.AIGPTSTORE,
                                                                          self.ctx.work_id)
        seed_edit_api = SeedEditAIGPTStoreAPI(platform_token=platform_token)

        # 获取图片尺寸
        img = cv2.imread(self.ctx.preprocessed_user_image_paths[0])
        height, width, _ = img.shape
        size = f"{width}x{height}"

        return asyncio.run(seed_edit_api.task_pipeline(
            model=model,
            image_url=self.ctx.oss_image_url_list[0],
            prompt=self.ctx.prompt,
            size=size,
            work_id=self.ctx.work_id,
            custom_request_type=custom_type
        ))


class Gpt4oAiGptStoreHandler(BaseImageEditHandler):
    def execute(self):
        is_openai_official = (self.ctx.sub_func == SubFuncType.GPT4O_IMAGE_OPENAI_AI_GPTSTORE.value)
        platform = ApiPlatformRedisKey.AIGPTSTORE_SSVIP if is_openai_official else ApiPlatformRedisKey.AIGPTSTORE
        platform_token = APIPlatformAccountManager.get_api_platform_token(platform, self.ctx.work_id)

        model = "gpt-image-1"  # 统一写死
        size = None if is_openai_official else self.ctx.params.get("gpt4o_size", '1024x1536')

        generation = ImageToImageProcessorEditAIAiGptStore(platform_token)
        return asyncio.run(generation.task_pipeline(
            work_id=self.ctx.work_id,
            prompt=self.ctx.prompt,
            model=model,
            quality=self.ctx.params.get("gpt4o_quality", "medium"),
            size=size,
            reference_image_paths=self.ctx.reference_image_paths,
            user_image_paths=self.ctx.preprocessed_user_image_paths,
            custom_request_type=self.ctx.sub_func
        ))


class Gpt4oNewApiHandler(BaseImageEditHandler):
    def execute(self):
        platform_token = APIPlatformAccountManager.get_api_platform_token(ApiPlatformRedisKey.NEWAPI, self.ctx.work_id)
        generation = ImageToImageProcessorChatNewApi(platform_token)
        return asyncio.run(generation.task_pipeline(
            work_id=self.ctx.work_id,
            prompt=self.ctx.prompt,
            model=self.ctx.params.get("gpt4o_model", "gpt-4o-image"),
            reference_image_paths=self.ctx.reference_image_paths,
            user_image_paths=self.ctx.preprocessed_user_image_paths,
            custom_request_type=SubFuncType.GPT4O_IMAGE_NEW_API
        ))


class RunwayHandler(BaseImageEditHandler):
    def execute(self):
        is_free = (self.ctx.sub_func == SubFuncType.FREE_RUNWAY_FRAMES_USEAPI.value)
        token = APIPlatformAccountManager.get_api_platform_token(ApiPlatformRedisKey.USEAPI, self.ctx.work_id)

        if is_free:
            account, password = RunwayAccountManager.get_runway_account_and_pwd_by_redis(
                account_type=RunwayAccountType.IMAGE_TO_IMAGE, work_id=self.ctx.work_id)
            # 稍微Hack一下，把账号信息放到结果里（原逻辑）
            self.ctx.final_result = RunwayAccountManager.append_runway_account_to_result(result={}, account=account)
            generator = ImageToVideoProcessorRunwayUserAPI(token, account, password)
            task = generator.frames_task_pipeline_with_retry(
                work_id=self.ctx.work_id,
                text_prompt=self.ctx.prompt,
                aspect_ratio=self.ctx.params.get("aspect_ratio", "16:9"),
                style=self.ctx.params.get("style"),
                num_images=int(self.ctx.params.get('num_images', 1)),
                user_image_paths=self.ctx.preprocessed_user_image_paths,
                reference_image_paths=self.ctx.reference_image_paths,
                explore_mode=False,
                custom_request_type=self.ctx.sub_func
            )
            return AsyncLoopManager.run_async_task(task)
        else:
            account = APIPlatformAccountManager.get_api_platform_token_with_round_robin(
                ApiPlatformRedisKey.USEAPI_RUNWAY, self.ctx.work_id)
            generator = ImageToVideoProcessorRunwayUserAPI(token, account)
            return asyncio.run(generator.frames_task_pipeline(
                work_id=self.ctx.work_id,
                text_prompt=self.ctx.prompt,
                aspect_ratio=self.ctx.params.get("aspect_ratio", "16:9"),
                style=self.ctx.params.get("style"),
                num_images=int(self.ctx.params.get('num_images', 1)),
                user_image_paths=self.ctx.preprocessed_user_image_paths,
                reference_image_paths=self.ctx.reference_image_paths,
                explore_mode=True,
                custom_request_type=self.ctx.sub_func
            ))


class LocalServiceHandler(BaseImageEditHandler):
    def execute(self):
        if self.ctx.sub_func == SubFuncType.LOCAL_KONTEXT_DEV.value:
            return kontetx_image_edit_function(
                work_id=self.ctx.work_id,
                file_path_list=self.ctx.preprocessed_user_image_paths,
                parameter_dict=self.ctx.params,
                logging=self.ctx.logging,
            )
        elif self.ctx.sub_func == SubFuncType.LOCAL_QWEN_EDIT.value:
            self.ctx.params['runninghub_run'] = self.ctx.input_dict.get("runninghub_run", None)
            return qwen_image_edit_function(
                work_id=self.ctx.work_id,
                file_path_list=self.ctx.preprocessed_user_image_paths,
                oss_image_url_list=self.ctx.oss_image_url_list,
                parameter_dict=self.ctx.params,
                logging=self.ctx.logging,
            )


class PixUseApiHandler(BaseImageEditHandler):
    def execute(self):
        token = APIPlatformAccountManager.get_api_platform_token(ApiPlatformRedisKey.USEAPI, self.ctx.work_id)
        account, password, account_level, _, _, premium_backup_flag, error = PixAccountManager.get_account_credentials(
            self.ctx.dynamic_params, None, None, self.ctx.work_id)

        # 暂存账号信息到上下文结果，以便后续合并
        result_container = {}
        PixAccountManager.update_pix_account_info(result_container, account_level, account)
        if error:
            result_container.update(error)
            return result_container  # 直接返回错误

        generator = ImageToVideoProcessorPixUseApi(token, account, password)
        task = generator.task_pipeline_with_retry(
            work_id=self.ctx.work_id,
            generation_type=GenerationTypeEnum.IMAGE_TO_IMAGE,
            custom_request_type=SubFuncType.PIX_IMG_TO_IMG_USEAPI,
            prompt=self.ctx.prompt,
            first_image_path=self.ctx.preprocessed_user_image_paths[0],
            template_id=self.ctx.params.get("template_id", ""),
            model="v5",
            quality="540p",
            duration=None,
            camera_movement=None,
            sound_effect_prompt=None,
            style=None,
            account_level=account_level
        )
        task_res = AsyncLoopManager.run_async_task(task)
        result_container["premium_backup_flag"] = premium_backup_flag
        result_container.update(task_res)
        return result_container


class GoogleFlowHandler(BaseImageEditHandler):
    def execute(self):
        platform_token = APIPlatformAccountManager.get_api_platform_token(ApiPlatformRedisKey.USEAPI, self.ctx.work_id)
        account_id = APIPlatformAccountManager.get_api_platform_token_with_round_robin(
            ApiPlatformRedisKey.USEAPI_GOOGLE_FLOW, self.ctx.work_id)
        generator = GoogleFlowUserAPI(token=platform_token, account_id=account_id)
        task = generator.image_edit_task_pipeline_with_retry(
            work_id=self.ctx.work_id,
            prompt=self.ctx.prompt,
            model=self.ctx.params.get("model_google_flow"),
            count=self.ctx.params.get("image_counts_google_flow", 1),
            aspect_ratio=self.ctx.params.get("aspect_ratio_google_flow"),
            user_image_paths=self.ctx.preprocessed_user_image_paths,
            reference_image_paths=self.ctx.reference_image_paths,
            custom_request_type=SubFuncType.GOOGLE_FLOW_USEAPI
        )
        return AsyncLoopManager.run_async_task(task)


class NanoBananaGptStoreHandler(BaseImageEditHandler):
    def execute(self):
        platform_token = APIPlatformAccountManager.get_api_platform_token(ApiPlatformRedisKey.AIGPTSTORE,
                                                                          self.ctx.work_id)
        generator = ImageEditNanoBananaAIAiGptStore(token=platform_token)

        # 判断是普通 Nano 还是 Gemini
        if self.ctx.sub_func == SubFuncType.GEMINI_AI_GPTSTORE.value:
            task = generator.gemini_task_pipeline(
                work_id=self.ctx.work_id,
                prompt=self.ctx.prompt,
                model="gemini-2.5-flash-image-preview",
                reference_image_paths=self.ctx.reference_image_paths,
                user_image_paths=self.ctx.preprocessed_user_image_paths,
                generation_type=GenerationTypeEnum.IMAGE_TO_IMAGE,
                custom_request_type=SubFuncType.GEMINI_AI_GPTSTORE
            )
        else:
            task = generator.task_pipeline(
                work_id=self.ctx.work_id,
                prompt=self.ctx.prompt,
                model=self.ctx.params.get("nano_banana_model", "nano-banana"),
                user_image_paths=self.ctx.preprocessed_user_image_paths,
                reference_image_paths=self.ctx.reference_image_paths,
                custom_request_type=SubFuncType.NANO_BANANA_AI_GPTSTORE
            )
        return AsyncLoopManager.run_async_task(task)


class NanoBananaFalAIHandler(BaseImageEditHandler):
    def execute(self):
        token = FalAIApi.get_redis_token(FalAiRedisKeyEnum.ACCOUNT_POOL, work_id=self.ctx.work_id)
        generator = ImageEditProcessorNanoBananaFalAI(platform_token=token)
        task = generator.task_pipeline_with_retry(
            work_id=self.ctx.work_id,
            prompt=self.ctx.prompt,
            num_images=self.ctx.params.get("nano_banana_num_images", 1),
            reference_image_paths=self.ctx.reference_image_paths,
            user_image_paths=self.ctx.preprocessed_user_image_paths,
            custom_request_type=SubFuncType.NANO_BANANA_FAL_AI
        )
        return AsyncLoopManager.run_async_task(task)


class NanoBananaReplicateHandler(BaseImageEditHandler):
    def execute(self):
        token = ReplicateAccountManager.get_api_platform_token_with_round_robin(self.ctx.work_id)
        generator = ImageEditProcessorNanoBananaReplicate(platform_token=token)
        task = generator.task_pipeline_with_retry(
            work_id=self.ctx.work_id,
            prompt=self.ctx.prompt,
            user_image_paths=self.ctx.preprocessed_user_image_paths,
            reference_image_paths=self.ctx.reference_image_paths,
            custom_request_type=SubFuncType.NANO_BANANA_REPLICATE
        )
        return AsyncLoopManager.run_async_task(task)


# Handler 工厂映射表
HANDLER_MAPPING = {
    SubFuncType.SEEDEDIT_OFFICIAL.value: SeedEditOfficialHandler,
    SubFuncType.SEEDEDIT_AI_GPTSTORE.value: SeedEditGptStoreHandler,
    SubFuncType.FLUX_KONTEXT_AI_GPTSTORE.value: SeedEditGptStoreHandler,
    SubFuncType.GPT4O_IMAGE_AI_GPTSTORE.value: Gpt4oAiGptStoreHandler,
    SubFuncType.GPT4O_IMAGE_OPENAI_AI_GPTSTORE.value: Gpt4oAiGptStoreHandler,
    SubFuncType.GPT4O_IMAGE_NEW_API.value: Gpt4oNewApiHandler,
    SubFuncType.RUNWAY_FRAMES_USEAPI.value: RunwayHandler,
    SubFuncType.FREE_RUNWAY_FRAMES_USEAPI.value: RunwayHandler,
    SubFuncType.LOCAL_KONTEXT_DEV.value: LocalServiceHandler,
    SubFuncType.LOCAL_QWEN_EDIT.value: LocalServiceHandler,
    SubFuncType.PIX_IMG_TO_IMG_USEAPI.value: PixUseApiHandler,
    SubFuncType.GOOGLE_FLOW_USEAPI.value: GoogleFlowHandler,
    SubFuncType.NANO_BANANA_AI_GPTSTORE.value: NanoBananaGptStoreHandler,
    SubFuncType.GEMINI_AI_GPTSTORE.value: NanoBananaGptStoreHandler,
    SubFuncType.NANO_BANANA_FAL_AI.value: NanoBananaFalAIHandler,  # 显式注册，用于降级调用
    SubFuncType.NANO_BANANA_REPLICATE.value: NanoBananaReplicateHandler,  # 显式注册，用于降级调用
}


def get_handler_class(sub_func_name):
    return HANDLER_MAPPING.get(str(sub_func_name))


# ==========================================
# 5. 辅助功能模块 (Helpers)
# ==========================================

def run_image_preprocess(ctx: ExecutionContext):
    """图片裁切与修复预处理"""
    recolor = ctx.params.get("recolor_switch", False)
    restored = ctx.params.get("face_restored_switch", False)
    img_processor = ImageProcessor(recolor_switch=recolor, face_restored_switch=restored)

    data = {
        "image_paths": ctx.input_path,
        "face_threshold": ctx.params.get('face_threshold', 0.3),
        "boundary_threshold": ctx.params.get('boundary_threshold', 20)
    }
    img_processor.set_strategy(ImageEditStrategy())
    result = img_processor.process_image(data)
    ctx.preprocessed_user_image_paths = result["image_paths"]


def run_oss_upload(ctx: ExecutionContext):
    """OSS 上传处理"""
    if ctx.sub_func in OSS_UPLOAD_REQUIRED_SUBFUNCS:
        for path in ctx.preprocessed_user_image_paths:
            res, url = upload_image_to_oss_cn(path)
            if res['flag'] != 1:
                return res  # Return failure dict
            ctx.oss_image_url_list.append(url)
    return None  # Success


def check_risk_control(ctx: ExecutionContext, output_dict):
    """风控检查"""
    # 定义各服务的风控正则
    RISK_MAP = [
        (APIContentRegexValidatorEnum.SEEDEDIT_API.value, RiskControlErrorCode.GENERATION_RESULT_VIOLATION.flag),
        (APIContentRegexValidatorEnum.GPT4O_API.value, RiskControlErrorCode.GENERATION_RESULT_VIOLATION.flag),
        (APIContentRegexValidatorEnum.RUNWAY_API.value, RiskControlErrorCode.GENERATION_RESULT_VIOLATION.flag),
        (APIContentRegexValidatorEnum.MANO_BANANA_API.value, RiskControlErrorCode.GENERATION_RESULT_VIOLATION.flag),
        (APIContentRegexValidatorEnum.MANO_BANANA_API_TEXT_VIOLATION.value,
         RiskControlErrorCode.SEXUAL_TEXT_VIOLATION.flag),
        (APIContentRegexValidatorEnum.RUNNINGHUB_API.value, RiskControlErrorCode.GENERATION_RESULT_VIOLATION.flag),
        (APIContentRegexValidatorEnum.GOOGLE_FLOW_MINOR_UPLOAD.value, RiskControlErrorCode.MINOR_IMAGE_VIOLATION.flag),
    ]

    error_types = [(flag, regex) for regex, flag in RISK_MAP]
    # 这里 platform 应该取最后成功的 platform，简化起见使用 default
    general_error_flag = PLATFORM_ERROR_CODES['default']

    return handle_video_generation_error_mapping(
        task_output=ctx.final_result,
        output_dict=output_dict,
        error_types=error_types,
        general_error_flag=general_error_flag
    )


def download_and_save_results(ctx: ExecutionContext, output_dict):
    """下载结果图片或处理本地路径"""
    result = ctx.final_result

    # 1. 处理直接返回本地路径的情况 (Local Services)
    if ctx.sub_func == SubFuncType.LOCAL_KONTEXT_DEV.value:
        ctx.upload_img_path_list = result.get('output', [])
        return True

    if ctx.sub_func == SubFuncType.LOCAL_QWEN_EDIT.value and result.get("local_comfyui_run", False):
        ctx.upload_img_path_list = result.get('output', [])
        return True

    # 2. 处理 URL 下载
    image_url_list = result.get('image_url', [])
    if not image_url_list:
        output_dict['flag'] = -1
        return False

    for image_url in image_url_list:
        image_name = generate_random_string(8) + ".jpg"
        save_path = os.path.join(CONFIG['oss_save_dir'], image_name)
        flag, msg = download_file_adapter(image_url, save_path)
        if not flag:
            output_dict['flag'] = -1
            output_dict['error_message'] = msg
            return False
        ctx.upload_img_path_list.append(save_path)

    # 3. Runway 特殊处理：补齐4张图
    if ctx.sub_func == SubFuncType.RUNWAY_FRAMES_USEAPI.value:
        while len(ctx.upload_img_path_list) < 4 and ctx.upload_img_path_list:
            ctx.upload_img_path_list.append(random.choice(ctx.upload_img_path_list))

    return len(ctx.upload_img_path_list) > 0


def process_grid_splitting(ctx: ExecutionContext):
    """宫格图片切分"""
    rows = int(ctx.params.get("img_rows", 1))
    cols = int(ctx.params.get("img_cols", 1))

    if rows > 1 or cols > 1:
        grid_list = []
        for path in ctx.upload_img_path_list:
            grid_list.extend(split_image_grid(path, rows, cols))
        ctx.upload_img_path_list = grid_list


def process_subtitles(ctx: ExecutionContext, output_dict):
    """字幕处理逻辑"""
    opt = ctx.params.get("subtitle_options", "disable")
    if opt not in ["enable_ai_generate", "enable_custom"] or not ctx.upload_img_path_list:
        return

    grid_num = int(ctx.params.get("grid_num_subtitle", 3))
    msg = "字幕生成成功"
    new_paths = []
    success_flag = True

    for img_path in ctx.upload_img_path_list:
        captions = None

        # AI 生成
        if opt == "enable_ai_generate":
            model = ctx.params.get("model_subtitle")
            prompt = ctx.params.get("prompt_subtitle")
            flag, content = generate_image_captions_with_llm(model, prompt, grid_num, ctx.work_id)
            if not flag:
                msg = f"图片{img_path}生成字幕失败: {content}"
                success_flag = False
                break
            captions = content

        # 自定义
        elif opt == "enable_custom":
            content = ctx.params.get("custom_content")
            if not content:
                msg = "自定义字幕为空"
                success_flag = False
                break
            valid, res = validate_custom_content(content, grid_num)
            if not valid:
                msg = f"自定义字幕格式错误: {res}"
                success_flag = False
                break
            captions = res

        # 合成字幕
        dir_name = os.path.dirname(img_path)
        ext = os.path.splitext(img_path)[1]
        out_path = os.path.join(dir_name, str(uuid.uuid4())[:8] + ext)

        if not add_captions_to_image(img_path, out_path, captions, grid_num):
            msg = "添加字幕失败，检查字体库"
            success_flag = False
            break
        new_paths.append(out_path)

    if success_flag:
        ctx.upload_img_path_list = new_paths

    output_dict['subtitle_message'] = msg


def send_dingtalk_alert(work_id, sub_func, current_handler_name, error_msg, prefix_msg):
    """发送钉钉告警"""
    body = {
        'workId': work_id,
        "ai": {"subFunc": sub_func},
    }
    generator = DingtalkAlertService()
    task = generator.send_dingtalk_alert(
        message=f"Nano-Banana API调用失败({prefix_msg} from {current_handler_name})",
        body=body,
        detail=error_msg,
        sender=SenderEnum.DING_AIGC.value,
    )
    AsyncLoopManager.run_async_task(task)


# ==========================================
# 6. 主入口函数 (Main Entry)
# ==========================================

def ImageEditFunction(input_dict: dict, logging):
    output_dict = {'flag': 1, 'pre_message': ""}
    ctx = ExecutionContext(input_dict, logging)

    logging.info(f"ImageEditWithSeedEditFunction input_dict: {input_dict}")

    try:
        # 1. 预处理 (Preprocess)
        run_image_preprocess(ctx)

        # 2. OSS 上传 (如有必要)
        oss_err = run_oss_upload(ctx)
        if oss_err: return oss_err

        # 3. 构建执行链 (Build Execution Chain)
        # 初始节点
        execution_chain = [{"target": ctx.sub_func, "alert": False}]
        # 获取降级配置并追加
        fallback_steps = FALLBACK_CONFIG.get(ctx.sub_func, FALLBACK_CONFIG["DEFAULT"])
        execution_chain.extend(fallback_steps)

        task_success = False

        # 4. 执行责任链 (Execute Chain)
        for i, step in enumerate(execution_chain):
            target_sub_func = step['target']
            alert_on_fail = step['alert']

            logging.info(f"[Chain Step {i + 1}] WorkID: {ctx.work_id} -> {target_sub_func}")

            HandlerCls = get_handler_class(target_sub_func)
            if not HandlerCls:
                logging.error(f"Handler missing for {target_sub_func}")
                continue

            try:
                handler = HandlerCls(ctx)
                result = handler.execute()

                need_fallback, err_msg = handler.should_fallback(result)

                if not need_fallback:
                    # === 成功 ===
                    task_success = True
                    ctx.final_result = result
                    # 如果有早期暂存在 result 里的账号信息等，这里会合并
                    if ctx.final_result:
                        output_dict.update(ctx.final_result)
                    # 如果有暂存在 ctx.final_result 外的特殊字段 (e.g. Pix)
                    if "premium_backup_flag" in result:
                        output_dict["premium_backup_flag"] = result["premium_backup_flag"]
                    break
                else:
                    # === 失败 ===
                    prefix = "has_fallback" if i < len(execution_chain) - 1 else "failed"
                    log_msg = f"[{target_sub_func} failed: {err_msg}] "
                    output_dict["pre_message"] += log_msg
                    logging.warning(f"Service {target_sub_func} failed. {err_msg}")

                    if alert_on_fail:
                        send_dingtalk_alert(ctx.work_id, ctx.sub_func, target_sub_func, err_msg, prefix)

            except Exception as e:
                logging.exception(f"Exception in {target_sub_func}: {e}")
                output_dict["pre_message"] += f"[{target_sub_func} Exception: {str(e)}] "
                if alert_on_fail:
                    send_dingtalk_alert(ctx.work_id, ctx.sub_func, target_sub_func, str(e), "exception")

        if not task_success:
            output_dict['flag'] = 0
            output_dict['error_message'] = f"Chain failed. Details: {output_dict.get('pre_message')}"
            return output_dict

        # 5. 风控 (Risk Control)
        risk_res = check_risk_control(ctx, output_dict)
        if risk_res: return risk_res

        # 6. 结果下载 (Download Results)
        download_success = download_and_save_results(ctx, output_dict)
        if not download_success:
            # download_and_save_results 内部已经设置了 error_message 和 flag=-1
            if output_dict.get('flag', 1) == 1:  # 如果还没设置失败
                output_dict['flag'] = -1
            return output_dict

        # 7. 宫格切分 (Grid Splitting)
        process_grid_splitting(ctx)

        # 8. 字幕处理 (Subtitles)
        process_subtitles(ctx, output_dict)

        # 9. 最终组装
        output_dict['upload_imgs_path'] = ctx.upload_img_path_list
        output_dict['processed_image'] = ctx.preprocessed_user_image_paths[0]

    except Exception as e:
        logging.exception(e)
        output_dict['flag'] = 0
        output_dict['error_message'] = "未知错误"

    finally:
        # 清理 OSS 资源
        for url in ctx.oss_image_url_list:
            try:
                delete_file_from_oss_cn_adapter(url)
            except:
                pass

        return output_dict