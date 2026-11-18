"""红方示例策略及其私有辅助实现。

assignment、rolling、coordination 与 learning 按策略类别提供文件入口；
common 保存这些示例复用的规划、执行和资源工具，deployment 负责生成初始位置。
选定入口文件同时导出 Deployment 和 RedAttack，供运行方分别完成部署与单机决策。
"""
