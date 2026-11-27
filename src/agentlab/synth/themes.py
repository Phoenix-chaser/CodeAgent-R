"""合成出题的主题：五个出题方向、一批具体领域，以及把"方向 × 领域"排成批次的办法。

题池只有 327 道题、难题只有 20 个来源族，RFT 看不出提升，根子在题太少。合成出题用教师接口成批地
出新题族，一个题族就是一个程序、一个来源族。这里只回答"向教师要一个什么样的题族"：一个主题是
一个出题方向（状态一致性、事务回滚、异常安全、锁序、条件变量）配上一个具体领域，模块名
s3_<slug>.py 和要求条数都由主题的 slug 决定。

排法是确定的：同样的参数永远给出同样的批次，中断后重跑、换机器续跑都对得上；已经出过题的主题
（used_slugs）不会再要一次。
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Direction:
    """一个出题方向。

    ``name`` 是短名（也是 slug 的前半段），``label`` 是中文名，``summary`` 是一句话说明，``slips``
    是这类题常见的疏忽（从现有难题的 mutant 里归纳）。``module`` 和 ``specs`` 指向现有难题里同一
    方向的题目列表，风格参照从那里挑。
    """

    name: str
    label: str
    summary: str
    slips: tuple[str, ...]
    module: str
    specs: str = "SPECS"


@dataclass(frozen=True)
class Domain:
    """一个具体领域：``name`` 是中文名（进提示），``code`` 是英文短名（进 slug），``hint`` 点出它涉及的数据和操作。"""

    name: str
    code: str
    hint: str


DIRECTIONS = {
    "state": Direction(
        name="state", label="状态一致性",
        summary="程序在主数据旁边维护一份辅助结构——二级索引、派生值、双向映射的另一半、登记表的状态和最近"
                "使用顺序——每一条修改路径都要让两者保持一致。",
        slips=(
            "辅助结构只在大路上同步：新增时两边都写了，改名、删除、批量更新、清空这些旁路上却只改了主数据，"
            "辅助结构里留下早已失效的条目。",
            "删掉一个键下最后一项时没有把空桶清掉，按键遍历的查询多出一个没有成员的键，这个键再次出现时的位置也不对。",
            "派生的合计、计数、最值在某一条修改路径上忘了作废，或者作废之后又把旧值写了回去，读到的是按修改前的"
            "数据算出的结果。",
            "先改后查：校验失败时异常照抛，但已经改掉的那一半留在原地，应当先校验再修改，"
            "或者在失败时把改过的部分还原。",
            "状态和顺序只改了一半：状态已经改成过期、吊销或淘汰，却没有把它移出最近使用列表，续期后也忘了挪到最新"
            "位置；到期判断的边界（now 等于 expires 算不算到期）容易写反。",
        ),
        module="agentlab.benchmark.hard_tasks_v3_state",
    ),
    "tx": Direction(
        name="tx", label="事务回滚",
        summary="一组操作要么全部生效、要么全部不生效：中间某一步失败时要把已经执行的步骤撤回，并把原来的异常"
                "原样抛出去。",
        slips=(
            "中途失败之后不回滚，或者只回滚了一部分：前面几步已经生效的修改留在原地，程序处在调用前和调用后各一半"
            "的状态。",
            "回滚的顺序不对：补偿操作按执行顺序而不是逆序进行，彼此重叠的修改（先改后删、同一项上的多次变动）"
            "抵消不掉。",
            "抛出的异常变了样：把原异常换成一句笼统的新异常、丢掉它的 __cause__ 或 __context__，或者让回滚中的小错误"
            "把原本要报告的异常盖住。",
            "成败的收尾没有做全：提交或回滚之后没有关闭，之后还能继续往里加；反过来失败之后却把工作单元关掉了；"
            "幂等键、分录序号这些成功的标记在成功之前就记上了。",
            "校验和计算的顺序颠倒：先改再查、边查边改，或者直接在原值上求值而不用副本，失败之后值、待重算集合这些"
            "与调用前不一致。",
        ),
        module="agentlab.benchmark.hard_tasks_v3_tx",
    ),
    "exc": Direction(
        name="exc", label="异常安全",
        summary="程序在正常路径上收尾是对的，缺陷只在某一步抛异常时暴露：清理被跳过或顺序错乱、原来的错误被后来的"
                "错误盖住、失败之后留下了不该留下的状态。",
        slips=(
            "失败路径上漏掉一步清理：只捕获了某一种异常，或者异常一出现就直接返回、直接抛出，忘了关句柄、删临时对象、"
            "释放资源。",
            "清理的顺序不对：回调和资源没有按登记的逆序收尾，或者先调用 close 再从池里移除（回调里再碰池时它还在），"
            "固定先后的步骤（先 flush 后 sync）被对调。",
            "后来的错误盖住了原来的错误：清理或回滚自己抛异常时，没有把它挂到原异常的 __context__ 或 __cause__ 上，"
            "或者反过来把原异常换成了清理的异常。",
            "关闭之后的门没关严：没有在一开始就置上已关闭标记，所以关闭过程中还能登记、重入的关闭会再跑一遍；嵌套时"
            "层数的增减与最外层的判断错开一位。",
            "重试与回滚配合出错：不判断异常能否重试就重试，最后一次尝试之前还多做了一次回滚，重试前的回滚失败之后"
            "没有停下来。",
        ),
        module="agentlab.benchmark.hard_tasks_v3_exc",
    ),
    "lock": Direction(
        name="lock", label="锁序",
        summary="程序会被多个线程同时使用，缺陷是每把锁各管一段解决不了的问题：按调用顺序加锁、复合操作先改后查、"
                "等锁超时被忽略、有的方法根本不加锁。",
        slips=(
            "按调用顺序加锁：一次要操作两个对象时按参数顺序去拿锁，方向相反的两个调用互相等待就死锁，应当按固定的"
            "全序（例如排好序的名字）加锁。",
            "复合操作只锁了一半：有的方法根本不加锁，或者先在锁外读、再在锁内写，校验放在修改之后，失败时已经改掉的"
            "那部分没有还原。",
            "等锁超时被忽略：acquire 返回 False 没有检查，或者等待超时之后仍按成功处理，调用方带着没拿到的锁"
            "继续往下走。",
            "状态变了却没有唤醒等待者：最后一个读者离开、写者释放、结果写入之后漏掉 notify_all，"
            "等待的线程一直睡到超时。",
            "等待和登记不对称：超时或失败的线程没有撤销自己的等待记录（等待中的写者个数、计算中的占位），后来的线程"
            "被一个早已离开的等待者挡住。",
        ),
        module="agentlab.benchmark.hard_tasks_v3_lock",
    ),
    "cond": Direction(
        name="cond", label="条件变量",
        summary="程序里有线程在等某个条件成立，或者在等别的线程产出结果；缺陷是被唤醒后不重新检查、关闭时漏掉一部分"
                "等待者、结果记给了错的调用方、停止时悄悄改变了排队中任务的去向。",
        slips=(
            "被唤醒之后不重新检查条件：把 while 写成 if，或者用 wait(timeout) 代替 wait_for(条件, timeout)，被别的"
            "线程抢先之后仍然往下执行。",
            "状态变了却没有唤醒：放入元素、写入结果、登记任务之后没有 notify_all，等待的线程要等到超时才醒来。",
            "关闭只通知了一部分等待者：close 只唤醒在同一个条件变量上等待的线程，另一类（等空位的放入、等结果的取出）"
            "一直阻塞。",
            "取出的顺序和归属出错：从队列尾部取、把结果记给错的票据、同一个结果交付两次、一个结果交给多个等待者。",
            "停止和超时改变了约定：等待超时返回 None 而不是抛 TimeoutError，停止时把还在排队的任务悄悄丢掉或者换成别的"
            "异常，任务抛出的异常被换成新的异常对象。",
        ),
        module="agentlab.benchmark.hard_tasks_v3_cond",
    ),
}
# 五个方向的固定顺序：按顺序取方向的地方（排批次、统计）都用它
DIRECTION_NAMES = tuple(DIRECTIONS)

# 具体领域：程序的主题。同一个方向换一个领域就是另一道题，所以名字要具体到能想出它管哪些数据、有哪些操作
DOMAINS = (
    Domain("库存", "inventory", "商品的数量、入库出库、盘点和预留"),
    Domain("账本", "ledger", "账户余额、复式分录和冲正"),
    Domain("缓存", "cache", "条目、容量、淘汰和失效"),
    Domain("任务队列", "task_queue", "任务的入队、优先级、领取和重试"),
    Domain("配置中心", "config_center", "配置项的版本、发布和回滚"),
    Domain("购物车", "shopping_cart", "商品行、数量修改、优惠和结算价"),
    Domain("日志轮转", "log_rotation", "内存里的日志段，按大小或条数轮换并保留若干份"),
    Domain("权限表", "permission_table", "角色、资源、授权和回收"),
    Domain("文档库", "document_store", "文档的登记、标签和检入检出"),
    Domain("日历", "calendar", "日程、重复规则、冲突和改期"),
    Domain("投票", "poll", "选项、票数、每人一票和截止"),
    Domain("成绩单", "gradebook", "学生、课程、成绩和加权平均"),
    Domain("订阅", "subscription", "套餐、计费周期、续订和暂停"),
    Domain("工单", "helpdesk_ticket", "工单的状态流转、指派和升级"),
    Domain("优惠券", "coupon", "领取、使用条件、核销和过期"),
    Domain("订单履约", "order_fulfillment", "拆单、发货、签收和库存扣减"),
    Domain("退款", "refund", "退款单、原路退回、部分退和失败重试"),
    Domain("发票", "invoice", "开票、红冲、额度和税率"),
    Domain("预算", "budget", "科目额度、占用、释放和结转"),
    Domain("报销", "expense_claim", "单据、审批、金额校验和打款"),
    Domain("考勤", "attendance", "打卡、迟到早退、补卡和请假冲抵"),
    Domain("排班", "roster", "班次、人员、换班和最少休息时间"),
    Domain("会议室预订", "room_booking", "时段、冲突、重复预订和取消"),
    Domain("图书借阅", "library_loan", "借出、归还、续借、预约和逾期"),
    Domain("选课", "course_enrollment", "课程容量、候补队列、退课和学分上限"),
    Domain("在线考试", "online_exam", "题目顺序、作答、交卷和计时"),
    Domain("作业提交", "homework_submission", "截止时间、迟交、版本和提交次数"),
    Domain("医院挂号", "clinic_registration", "号源、科室、预约和退号"),
    Domain("药房发药", "pharmacy_dispensary", "处方、库存扣减、配伍禁忌和发药"),
    Domain("酒店入住", "hotel_checkin", "房型、入住退房、换房和锁房"),
    Domain("航班值机", "flight_checkin", "座位、舱位、登机牌和超售"),
    Domain("火车票改签", "train_rebooking", "余票、改签、差价和退票费"),
    Domain("租车", "car_rental", "车辆、租期、取还车和押金"),
    Domain("景点门票", "attraction_ticket", "分时段票额、核销和退票"),
    Domain("餐厅等位", "restaurant_waitlist", "取号、叫号、过号和排队优先"),
    Domain("外卖配送", "delivery_dispatch", "订单派单、骑手取餐、送达和超时"),
    Domain("食材保质期", "shelf_life", "批次、到期日、先进先出和报损"),
    Domain("健身房会员", "gym_membership", "会籍、冻结、续费和课位预约"),
    Domain("比赛计分板", "scoreboard", "得分事件、比分、暂停和撤销"),
    Domain("锦标赛赛程", "tournament_schedule", "轮次、对阵、轮空和晋级"),
    Domain("停车场", "parking_lot", "车位、进出、计费和满位"),
    Domain("充电桩", "charging_station", "充电口、预约、功率分配和结算"),
    Domain("电梯调度", "elevator_dispatch", "呼梯、停靠层、顺路搭载和限载"),
    Domain("智能家居场景", "home_scene", "设备、触发条件、联动动作和冲突"),
    Domain("传感器读数", "sensor_reading", "上报、去重、窗口聚合和异常标记"),
    Domain("门禁卡", "access_badge", "发卡、授权区域、刷闸和挂失"),
    Domain("快递柜", "parcel_locker", "格口、投递、取件码和超时滞留"),
    Domain("仓库拣货", "warehouse_picking", "拣货单、波次、库位和缺货"),
    Domain("集装箱装船", "container_loading", "舱位、配载、重量限制和卸港顺序"),
    Domain("海关申报", "customs_declaration", "商品归类、限额、查验和放行"),
    Domain("补货", "replenishment", "在途、安全库存、再订货点和到货"),
    Domain("质检批次", "inspection_batch", "抽样、判定、批号和召回"),
    Domain("设备维保", "maintenance_schedule", "保养周期、工单、备件和逾期"),
    Domain("备件库", "spare_parts", "领用、归还、以旧换新和再订货点"),
    Domain("农田灌溉", "irrigation", "地块、水量配额、轮灌时段和阀门"),
    Domain("温室环境", "greenhouse_control", "温湿度区间、通风、加湿和告警"),
    Domain("天气预警", "weather_alert", "预警级别、生效时段、订阅和解除"),
    Domain("河流水位", "river_gauge", "测站、水位、警戒线和涨落趋势"),
    Domain("电力负荷", "power_load", "峰谷时段、负荷上限、错峰和优先级"),
    Domain("储罐液位", "tank_gauge", "液位、进出流量、上下限和混油"),
    Domain("光伏发电", "solar_output", "逆变器、发电量、限功率和日汇总"),
    Domain("出租车计价", "taxi_meter", "起步价、里程、低速等候和夜间加价"),
    Domain("公交到站", "bus_arrival", "班次、站点、到站预报和晚点"),
    Domain("共享单车", "bike_share", "车辆、解锁、骑行、还车和调度区"),
    Domain("网约车派单", "ride_hailing", "乘客、司机、距离、应答和取消"),
    Domain("船舶泊位", "berth_allocation", "泊位、靠离港时间、长度和移泊"),
    Domain("用户会话", "user_session", "登录、令牌、到期、续期和吊销"),
    Domain("接口限流", "rate_limiter", "配额窗口、突发、排队和拒绝"),
    Domain("熔断器", "circuit_breaker", "失败计数、半开、恢复和降级"),
    Domain("服务注册表", "service_registry", "实例、心跳、摘除和恢复"),
    Domain("消息主题", "message_topic", "订阅、投递、确认和重投"),
    Domain("版本发布", "release_train", "分支、版本号、冻结期和回退"),
    Domain("功能开关", "feature_flag", "规则、灰度比例、人群和开关状态"),
    Domain("灰度发布", "canary_rollout", "阶段、放量、指标门槛和回滚"),
    Domain("密钥轮换", "key_rotation", "密钥版本、生效、重叠期和作废"),
    Domain("审计日志", "audit_trail", "事件序列、完整性校验和留存"),
    Domain("告警抑制", "alert_dedup", "告警指纹、抑制窗口、分组和恢复"),
    Domain("监控指标", "metrics_rollup", "采样、时间窗口、降采样和聚合"),
    Domain("延时任务", "delayed_job", "到期时刻、优先队列、取消和执行"),
    Domain("工作流引擎", "workflow_engine", "步骤、依赖、分支和失败重试"),
    Domain("审批流", "approval_chain", "审批人、会签或签、转交和驳回"),
    Domain("电子签名", "signature_request", "签署方、顺序、催办和作废"),
    Domain("合同续约", "contract_renewal", "到期、续约条款、提醒和终止"),
    Domain("股权台账", "cap_table", "股东、股份变动、稀释和轮次"),
    Domain("证券撮合", "order_book", "买卖盘、价格优先、成交和撤单"),
    Domain("众筹", "crowd_funding", "支持金额、目标、进度和失败退款"),
    Domain("拍卖竞价", "auction", "出价、领先价、保留价和延时截止"),
    Domain("保险保单", "insurance_policy", "保额、保费、生效、批改和退保"),
    Domain("理赔", "claim_processing", "报案、定损、免赔额和赔付状态"),
    Domain("会员等级", "loyalty_tier", "积分、升降级、保级期和权益"),
    Domain("里程账户", "mileage_account", "里程累积、兑换、过期和冻结"),
    Domain("礼品卡", "gift_card", "面值、充值、消费和余额"),
    Domain("拼团", "group_buying", "成团人数、参团、超时未成团退款"),
    Domain("秒杀", "flash_sale", "活动库存、限购、下单和回补"),
    Domain("宿舍分配", "dormitory_assignment", "床位、室友约束、调宿和退宿"),
    Domain("奖学金", "scholarship", "评定条件、名额、排名和公示"),
    Domain("游戏匹配", "matchmaking", "段位、等待时间、房间和匹配池"),
    Domain("排行榜", "leaderboard", "分数、排名、并列和赛季重置"),
    Domain("聊天室", "chat_room", "成员、消息顺序、禁言和撤回"),
    Domain("待办清单", "todo_list", "事项、截止、优先级、完成和递延"),
    Domain("看板", "kanban_board", "列、卡片、在制品限制和移动"),
    Domain("代码评审", "code_review", "评审人、评论、批准、驳回和合并条件"),
    Domain("实验室样本", "lab_sample", "样本编号、流转记录、存放位置和销毁"),
    Domain("救灾物资", "disaster_relief", "物资、需求点、调拨和优先级"),
    Domain("血库", "blood_bank", "血型、单位、有效期和交叉配血"),
    Domain("失物招领", "lost_and_found", "物品登记、认领核验、招领和上交"),
    Domain("法庭排期", "court_docket", "法庭、庭审、时长、冲突和改期"),
    Domain("地块登记", "land_parcel", "地块、权利人、转移、抵押和查封"),
    Domain("房产挂牌", "property_listing", "挂牌、带看、议价、成交和下架"),
)
DOMAIN_NAMES = tuple(domain.name for domain in DOMAINS)

# slug 进了模块名、task_id 和文件名，只能用这些字符
SLUG_PATTERN = re.compile(r"[a-z0-9_]+")
_DOMAINS_BY_CODE = {domain.code: domain for domain in DOMAINS}
_DOMAINS_BY_NAME = {domain.name: domain for domain in DOMAINS}


def resolve_direction(value) -> Direction:
    """方向对象，或者它的短名（state）、中文名（状态一致性）→ 方向对象。认不出抛 ValueError。"""
    if isinstance(value, Direction):
        return value
    if isinstance(value, str) and value in DIRECTIONS:
        return DIRECTIONS[value]
    for direction in DIRECTIONS.values():
        if direction.label == value:
            return direction
    known = "、".join(f"{item.name}（{item.label}）" for item in DIRECTIONS.values())
    raise ValueError(f"未知的方向 {value!r}：可选的是 {known}")


def resolve_domain(value) -> Domain:
    """领域对象，或者它的英文短名、中文名 → 领域对象。认不出抛 ValueError。"""
    if isinstance(value, Domain):
        return value
    if isinstance(value, str):
        if value in _DOMAINS_BY_CODE:
            return _DOMAINS_BY_CODE[value]
        if value in _DOMAINS_BY_NAME:
            return _DOMAINS_BY_NAME[value]
    raise ValueError(f"未知的领域 {value!r}：要在 themes.DOMAINS 里，或者自己给一个 Domain 对象")


def resolve_theme(theme) -> tuple[Direction, Domain]:
    """``(方向, 领域)``：两个元素都可以是对象，也可以是名字。"""
    try:
        direction, domain = theme
    except (TypeError, ValueError):
        raise ValueError(f"主题应当是 (方向, 领域) 二元组，现在是 {theme!r}") from None
    return resolve_direction(direction), resolve_domain(domain)


def slug_of(direction_name, domain) -> str:
    """主题的 slug：``方向短名_领域短名``，例如 state_inventory。它进模块名、task_id 和 source_group。"""
    direction = resolve_direction(direction_name)
    target = resolve_domain(domain)
    slug = f"{direction.name}_{target.code}"
    if not SLUG_PATTERN.fullmatch(slug):
        raise ValueError(f"slug 只能用小写字母、数字、下划线：{slug!r}")
    return slug


def _sequence(seed: int) -> tuple[tuple[Direction, Domain], ...]:
    """全部"方向 × 领域"对排成的一条固定序列。

    先给每个方向把领域表各洗一次牌，再一轮一轮地取：第 r 轮从每个方向洗好的表里取第 r 个领域，把这 5 个
    对再洗一次牌接在序列后面。所以序列里每连续 5 个位置上五个方向各出现一次，一批 50 个主题就是每个方向
    10 个，不会某一批偏重某个方向。只依赖 random.Random(seed)，同样的种子永远得到同一条序列。
    """
    rng = random.Random(seed)
    shuffled = []
    for _ in DIRECTIONS:
        order = list(DOMAINS)
        rng.shuffle(order)
        shuffled.append(order)
    pairs = []
    for round_index in range(len(DOMAINS)):
        block = [(direction, shuffled[index][round_index])
                 for index, direction in enumerate(DIRECTIONS.values())]
        rng.shuffle(block)
        pairs.extend(block)
    return tuple(pairs)


def plan_themes(batch, families, *, seed=42, used_slugs=()) -> list[tuple[Direction, Domain]]:
    """第 ``batch`` 批（从 1 起）的 ``families`` 个主题，每个是 ``(方向, 领域)``。

    主题取自 ``_sequence(seed)`` 这条固定序列，从第 (batch-1)*families 个开始，slug 在 ``used_slugs`` 里的
    跳过，凑够 ``families`` 个为止。同样的参数永远给同样的结果；序列不够长时抛 ValueError。
    """
    if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
        raise ValueError(f"batch 从 1 起，现在是 {batch!r}")
    if not isinstance(families, int) or isinstance(families, bool) or families < 1:
        raise ValueError(f"families 至少是 1，现在是 {families!r}")
    taken = frozenset(used_slugs)
    start = (batch - 1) * families
    plan = []
    for direction, domain in _sequence(seed)[start:]:
        if len(plan) == families:
            return plan
        if f"{direction.name}_{domain.code}" in taken:
            continue
        plan.append((direction, domain))
    if len(plan) < families:                      # 序列走到头了还不够（正好够的时候循环里已经返回）
        raise ValueError(f"第 {batch} 批凑不够 {families} 个主题：序列从第 {start + 1} 个起到末尾、"
                         f"去掉 {len(taken)} 个已经用过的 slug，只剩 {len(plan)} 个")
    return plan


def stable_hash(text: str) -> int:
    """由字符串决定的整数，跨进程、跨 Python 版本都不变。

    "由 slug 决定、结果确定"的选择都用它：内置的 hash() 对字符串每个进程都加了随机盐，同一分钟内两次调用
    都可能不一样。
    """
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


# 一道题有几条编号的行为要求：由 slug 决定，在这个范围里。放在这里是因为 parse（把关条数）和 prompts
# （把条数写进提示）都要用它，而 prompts 依赖 parse，不能反过来导入
MIN_REQUIREMENTS, MAX_REQUIREMENTS = 4, 6


def requirement_count(slug) -> int:
    """这道题要几条行为要求：由 slug 决定，在 MIN_REQUIREMENTS 到 MAX_REQUIREMENTS 之间；同一个 slug 永远是同一个数。"""
    return MIN_REQUIREMENTS + stable_hash(f"requirements:{slug}") % (MAX_REQUIREMENTS - MIN_REQUIREMENTS + 1)


__all__ = ["DIRECTIONS", "DIRECTION_NAMES", "DOMAINS", "DOMAIN_NAMES", "MAX_REQUIREMENTS", "MIN_REQUIREMENTS",
           "SLUG_PATTERN", "Direction", "Domain", "plan_themes", "requirement_count", "resolve_direction",
           "resolve_domain", "resolve_theme", "slug_of", "stable_hash"]
