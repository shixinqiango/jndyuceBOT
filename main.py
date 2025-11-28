import asyncio
import json
import os
import requests
import numpy as np # 需要安装 numpy: pip install numpy
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ==================== 基础配置 ====================
BOT_TOKEN = '你的_BOT_TOKEN_在这里'
API_URL = "https://pc28.help/kj.json?limit=200" # 获取更多数据以建立矩阵
DATA_FILE = "groups.json"
SHOW_LIMIT = 8

# ==================== 🧮 高级算法核心 ====================

class AdvancedAlgo:
    def __init__(self):
        self.options = ["大单", "大双", "小单", "小双"]

    def get_type(self, num_str):
        """解析数字属性"""
        try:
            n = int(num_str)
            is_big = n >= 14
            is_odd = n % 2 != 0
            if is_big and is_odd: return "大单"
            if is_big and not is_odd: return "大双"
            if not is_big and is_odd: return "小单"
            if not is_big and not is_odd: return "小双"
        except:
            return None
        return None

    def calculate_markov_kill(self, data_list):
        """
        【核心算法1：马尔可夫链状态转移矩阵】
        计算：基于上一期结果，下一期跳到哪个组合的概率最低？
        """
        # 1. 数据清洗，转为类型列表 [大单, 小双, 大单, ...]
        history_types = []
        # 注意：API返回通常是倒序的(最新在0)，我们需要正序(从旧到新)来建立链条
        sorted_data = sorted(data_list, key=lambda x: x['qihao'])
        
        for item in sorted_data:
            t = self.get_type(item['sum'])
            if t: history_types.append(t)

        if len(history_types) < 10: return None # 数据不够

        # 2. 建立转移矩阵
        # 结构: { "大单": {"大单":0, "大双":0...}, "小双": {...} }
        matrix = {k: {o: 0 for o in self.options} for k in self.options}
        
        # 统计 A -> B 的次数
        for i in range(len(history_types) - 1):
            current = history_types[i]
            next_one = history_types[i+1]
            matrix[current][next_one] += 1

        # 3. 获取最后一期结果（当前状态）
        last_val = history_types[-1]
        
        # 4. 分析当前状态的后续概率
        transitions = matrix[last_val] 
        # 例如: 上期是小双。
        # 历史显示接大单5次，接大双20次，接小单15次，接小双3次。
        # 那么接“小双”概率最低（只有3次）。
        
        # 按出现次数排序 (从小到大)
        sorted_trans = sorted(transitions.items(), key=lambda x: x[1])
        
        # 返回次数最少的那个（即预测最不可能出现的 -> 杀它）
        kill_target = sorted_trans[0][0]
        
        # 打印日志方便调试
        print(f"🧬 马尔可夫分析: 上期[{last_val}] -> 历史后续分布 {transitions} -> 推荐杀: {kill_target}")
        return kill_target

    def calculate_ema_kill(self, data_list):
        """
        【核心算法2：EMA 指数平滑移动平均】
        计算趋势分值，近期出现的权重极高。
        杀掉分数最低（近期走势最弱）的组合。
        """
        scores = {k: 0.0 for k in self.options}
        alpha = 0.2 # 平滑系数，越大数据越敏感
        
        # 正序遍历
        sorted_data = sorted(data_list, key=lambda x: x['qihao'])
        
        for item in sorted_data:
            t = self.get_type(item['sum'])
            if not t: continue
            
            # 每一期，命中的组合分数增加，其他的衰减
            for k in self.options:
                if k == t:
                    # 命中：EMA = alpha * 1 + (1-alpha) * old
                    scores[k] = alpha * 1.0 + (1 - alpha) * scores[k]
                else:
                    # 没中：EMA = alpha * 0 + (1-alpha) * old
                    scores[k] = (1 - alpha) * scores[k]
        
        # 找出分数最低的（最冷/趋势最差）
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        kill_target = sorted_scores[0][0]
        
        print(f"📉 EMA趋势分析: 分数分布 {scores} -> 推荐杀: {kill_target}")
        return kill_target

    def get_prediction(self, data_list):
        """
        【双核决策系统】
        优先使用马尔可夫，如果数据不足或异常，使用EMA趋势。
        """
        try:
            # 优先尝试马尔可夫
            pred = self.calculate_markov_kill(data_list)
            if pred:
                return pred
            
            # 兜底使用 EMA
            return self.calculate_ema_kill(data_list)
        except Exception as e:
            print(f"算法出错: {e}")
            return "小双" # 终极兜底

# ==================== 机器人逻辑 ====================

class Manager:
    def __init__(self):
        self.algo = AdvancedAlgo()
        self.chats = self.load_chats()
        self.history = [] 
        self.last_qihao = 0
        self.next_kill = None

    def load_chats(self):
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r') as f: return set(json.load(f))
        return set()

    def save_chats(self):
        with open(DATA_FILE, 'w') as f: json.dump(list(self.chats), f)

    def add_chat(self, cid):
        if cid not in self.chats:
            self.chats.add(cid)
            self.save_chats()
            return True
        return False

    def fetch_data(self):
        try:
            # 获取200期以保证矩阵准确
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(API_URL, headers=headers, timeout=10)
            data = resp.json()
            if 'data' in data: return data['data']
        except Exception as e:
            print(f"网络错误: {e}")
        return None

    def build_msg(self, curr_qihao):
        msg = ""
        start = max(0, len(self.history) - SHOW_LIMIT)
        for row in self.history[start:]:
            mark = "✅" if row['win'] else "❌"
            msg += f"{row['qihao']}期 预测杀组➜ 杀{row['pred']} {mark}\n"
            
        nxt = int(curr_qihao) + 1
        msg += f"{nxt}期 预测杀组➜ 杀{self.next_kill}"
        return msg

manager = Manager()

# ==================== TG Handlers ====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    if manager.add_chat(cid):
        await update.message.reply_text("✅ 已连接加拿大28核心数据库。\n已启用 [Markov链] + [EMA趋势] 双核算法。")

async def loop_monitor(app: Application):
    print("🚀 高级算法引擎已启动...")
    
    # 第一次初始化
    d = manager.fetch_data()
    if d:
        manager.last_qihao = int(d[0]['qihao'])
        manager.next_kill = manager.algo.get_prediction(d)
        print(f"初始化预测: {manager.next_kill}")

    while True:
        try:
            raw_data = manager.fetch_data()
            if raw_data:
                latest = raw_data[0]
                curr_q = int(latest['qihao'])
                curr_sum = latest['sum']

                if curr_q > manager.last_qihao:
                    print(f"\n★ 新开奖: {curr_q}期 -> {curr_sum}")
                    
                    # 结算
                    actual = manager.algo.get_type(curr_sum)
                    is_win = False
                    if manager.next_kill:
                        # 杀A，开B = 赢
                        is_win = (manager.next_kill != actual)
                        
                        manager.history.append({
                            'qihao': curr_q,
                            'pred': manager.next_kill,
                            'win': is_win
                        })
                    
                    # 计算下期
                    new_pred = manager.algo.get_prediction(raw_data)
                    manager.next_kill = new_pred
                    
                    # 发送
                    txt = manager.build_msg(curr_q)
                    for cid in list(manager.chats):
                        try:
                            await app.bot.send_message(cid, txt)
                        except: pass
                    
                    manager.last_qihao = curr_q
                    
            await asyncio.sleep(8)
        except Exception as e:
            print(f"Loop Error: {e}")
            await asyncio.sleep(5)

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    loop = asyncio.get_event_loop()
    loop.create_task(loop_monitor(app))
    app.run_polling()

if __name__ == "__main__":
    main()
