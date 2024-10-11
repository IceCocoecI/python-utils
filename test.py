import copy

# 原始列表
original_list = [1, 2, [3, 4]]

# 浅拷贝
shallow_copied_list = copy.copy(original_list)

print("原始列表：", original_list)
print("浅拷贝列表：", shallow_copied_list)

# 修改原始列表中的嵌套列表
original_list[1] = 5

print("修改后原始列表：", original_list)
print("修改后浅拷贝列表：", shallow_copied_list)