import numpy as np
import tsplib95

class TSPSpecification:
    """定义 TSP 问题的输入格式和解的表示方式"""

    def __init__(self, num_cities: int, distances: np.ndarray):
        self.num_cities = num_cities
        self.distances = distances

    def is_valid_solution(self, tour: list) -> bool:
        """检查解是否有效（访问所有城市且不重复）"""
        return set(tour) == set(range(self.num_cities)) and len(tour) == self.num_cities

    def objective_function(self, tour: list) -> float:
        """计算路径的总距离"""
        total_distance = sum(self.distances[tour[i], tour[i + 1]] for i in range(len(tour) - 1))
        total_distance += self.distances[tour[-1], tour[0]]  # 回到起点
        return total_distance

    def random_solution(self):
        """随机生成一个有效路径"""
        tour = list(range(self.num_cities))
        np.random.shuffle(tour)
        return tour

def evaluate_tsp(tour: list, tsp_spec: TSPSpecification) -> float:
    """计算旅行商路径的总长度"""
    if not tsp_spec.is_valid_solution(tour):
        return float('inf')  # 如果解无效，给个极大的代价
    return tsp_spec.objective_function(tour)

def mutate_tour(tour: list) -> list:
    """对路径进行随机变异，交换两个城市的位置"""
    new_tour = tour[:]
    i, j = np.random.choice(len(tour), 2, replace=False)  # 随机选两个城市
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]   # 交换它们
    return new_tour


# def load_tsplib_problem(filepath: str):
#     """加载 TSPLib 数据集并转换为距离矩阵"""
#     problem = tsplib95.load(filepath)
#
#     num_cities = len(list(problem.get_nodes()))  # 获取城市数量
#     distances = np.zeros((num_cities, num_cities))
#
#     for i in range(num_cities):
#         for j in range(num_cities):
#             if i != j:
#                 distances[i, j] = problem.get_weight(i + 1, j + 1)  # TSPLib 从 1 开始索引
#
#     return num_cities, distances

def load_tsplib_problem(filepath):
    problem = tsplib95.load(filepath)
    num_cities = problem.dimension  # 获取城市数目
    distances = np.zeros((num_cities, num_cities))  # 创建距离矩阵

    # 填充距离矩阵
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distances[i, j] = problem.get_weight(i + 1, j + 1)  # TSPLIB 索引从 1 开始

    return num_cities, distances.tolist()  # 返回普通 Python 结构，避免序列化问题


# 计算距离矩阵
def compute_distance_matrix(problem):
    coords = np.array([problem.node_coords[i + 1] for i in range(problem.dimension)])
    num_cities = len(coords)
    distances = np.zeros((num_cities, num_cities))

    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distances[i, j] = np.linalg.norm(coords[i] - coords[j])

    return distances
