from typing import Generic, TypeVar, Optional, List, Literal

from pydantic import Field, BaseModel, model_validator

# 定义一个泛型类型 T
T = TypeVar('T')

# 定义 Status 的字面量类型，增强类型安全
Status = Literal['success', 'failed']

from typing import Generic, TypeVar, Optional, List, Literal
from pydantic import BaseModel, Field, model_validator

T = TypeVar('T')
Status = Literal['success', 'failed']


class BaseResponse(BaseModel, Generic[T]):
    """
    增加了 is_success 和 is_failed 属性的基础响应结构
    """
    code: int = Field(default=1, description="业务状态码, 1 表示成功")
    msg: str = Field(default="success", description="响应消息")
    status: Status = Field(default='success', description="请求状态: 'success' 或 'failed'")
    data: Optional[T] = Field(default=None, description="响应数据")

    @model_validator(mode='after')
    def check_code_and_status(self) -> 'BaseResponse':
        is_success = self.code == 1
        expected_status: Status = 'success' if is_success else 'failed'

        if self.status != expected_status:
            self.status = expected_status

        if not is_success and self.msg == "success":
            self.msg = "request failed"

        return self

    # --- 新增的属性 ---
    @property
    def is_success(self) -> bool:
        """
        判断请求是否成功
        """
        return self.code == 1

    @property
    def is_failed(self) -> bool:
        """
        判断请求是否失败
        """
        return not self.is_success

    # --- --- --- ---

    @classmethod
    def success(cls, data: T = None, msg: str = "success") -> 'BaseResponse[T]':
        return cls(code=1, msg=msg, data=data, status='success')

    @classmethod
    def error(cls, code: int, msg: str) -> 'BaseResponse':
        if code == 1:
            raise ValueError("Error response code cannot be 1.")
        return cls(code=code, msg=msg, data=None)



# 首先定义一个可复用的、带上下文元数据的响应类
class ApiResponse(BaseResponse[T], Generic[T]):
    request_url: Optional[str] = None
    request_account: Optional[str] = None





if __name__ == '__main__':
    # 创建一个成功的响应
    success_response = BaseResponse.success(data={"name": "John Doe", "age": 30})
    data = success_response.data
    print(success_response.model_dump_json())
    print(data)

    # 创建一个失败的响应
    error_response = BaseResponse.error(code=404, msg="Not Found")
    print(error_response.model_dump_json())

    # 使用时
    response = ApiResponse(
        request_url="/users/1",
        request_account="admin"
    )

    print(response.model_dump_json())

    response = ApiResponse.error(code=404, msg="Not Found")
    print(response.model_dump_json())

