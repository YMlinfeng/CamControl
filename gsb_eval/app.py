# import os
# import random
# import sqlite3
# from flask import Flask, render_template, request, jsonify, Response
# import csv
# from io import StringIO

# app = Flask(__name__)

# # 配置视频文件夹路径
# FOLDER_A = 'static/folder_a'
# FOLDER_B = 'static/folder_b'
# DB_PATH = 'eval_results.db'

# def init_db():
#     """初始化数据库表，记录配对的索引和双方的视频名"""
#     with sqlite3.connect(DB_PATH) as conn:
#         cursor = conn.cursor()
#         cursor.execute('''
#             CREATE TABLE IF NOT EXISTS evaluations (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 pair_index INTEGER,
#                 video_a TEXT,
#                 video_b TEXT,
#                 evaluator_id TEXT,
#                 camera_motion TEXT,
#                 image_quality TEXT,
#                 dynamic_rationality TEXT,
#                 aesthetics TEXT,
#                 overall TEXT,
#                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
#             )
#         ''')
#         conn.commit()

# init_db()

# def get_video_pairs():
#     """将两个文件夹的视频按名称排序，并顺次一对一配对"""
#     # 获取并排序所有 mp4 文件
#     videos_a = sorted([f for f in os.listdir(FOLDER_A) if f.endswith('.mp4')])
#     videos_b = sorted([f for f in os.listdir(FOLDER_B) if f.endswith('.mp4')])
    
#     # zip() 函数会自动以最短的列表为准进行截断，完美满足"以一个文件夹取完为准"
#     pairs = list(zip(videos_a, videos_b))
#     return pairs

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/api/get_task', methods=['GET'])
# def get_task():
#     pairs = get_video_pairs()
#     if not pairs:
#         return jsonify({"error": "文件夹中没有找到视频，请检查路径"}), 404
        
#     # 查询数据库中已经评测过的视频对索引 (pair_index)
#     with sqlite3.connect(DB_PATH) as conn:
#         cursor = conn.cursor()
#         cursor.execute('SELECT DISTINCT pair_index FROM evaluations')
#         evaluated_indices = {row[0] for row in cursor.fetchall()}
        
#     # 过滤出还没有被评测过的视频对索引
#     pending_indices = [i for i in range(len(pairs)) if i not in evaluated_indices]
    
#     if not pending_indices:
#         return jsonify({"error": "🎉 所有视频对已全部评测完毕！"}), 404
        
#     # 为了防止并发时多个人拿到同一个任务，从剩余任务中随机抽取一个分配
#     # pair_index = random.choice(pending_indices) #!
#     # 单人评测，直接按顺序取第一个未评测的
#     pair_index = pending_indices[0]
#     video_a, video_b = pairs[pair_index]
    
#     # 随机打乱左右顺序，实现盲测 (不让评测员知道哪个是A哪个是B)
#     is_a_left = random.choice([True, False])
    
#     if is_a_left:
#         left_url = f"{FOLDER_A}/{video_a}"
#         right_url = f"{FOLDER_B}/{video_b}"
#     else:
#         left_url = f"{FOLDER_B}/{video_b}"
#         right_url = f"{FOLDER_A}/{video_a}"
        
#     return jsonify({
#         "pair_index": pair_index,
#         "video_a": video_a,
#         "video_b": video_b,
#         "left_url": left_url,
#         "right_url": right_url,
#         "is_a_left": is_a_left 
#     })

# @app.route('/api/submit', methods=['POST'])
# def submit():
#     data = request.json
#     pair_index = data.get('pair_index')
#     video_a = data.get('video_a')
#     video_b = data.get('video_b')
#     is_a_left = data.get('is_a_left')
#     metrics = data.get('metrics')
    
#     # 将前端的 Left/Right/Tie 映射回 A/B/Tie
#     def map_result(choice):
#         if choice == 'Tie': return 'Tie'
#         if choice == 'Left': return 'A' if is_a_left else 'B'
#         if choice == 'Right': return 'B' if is_a_left else 'A'
#         return 'Unknown'
    
#     mapped_metrics = {k: map_result(v) for k, v in metrics.items()}
    
#     with sqlite3.connect(DB_PATH, timeout=10) as conn:
#         cursor = conn.cursor()
#         cursor.execute('''
#             INSERT INTO evaluations 
#             (pair_index, video_a, video_b, evaluator_id, camera_motion, image_quality, dynamic_rationality, aesthetics, overall)
#             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
#         ''', (
#             pair_index,
#             video_a,
#             video_b,
#             request.remote_addr, # 记录评测员IP
#             mapped_metrics.get('camera_motion'),
#             mapped_metrics.get('image_quality'),
#             mapped_metrics.get('dynamic_rationality'),
#             mapped_metrics.get('aesthetics'),
#             mapped_metrics.get('overall')
#         ))
#         conn.commit()
        
#     return jsonify({"status": "success"})

# @app.route('/export')
# def export_csv():
#     with sqlite3.connect(DB_PATH) as conn:
#         cursor = conn.cursor()
#         cursor.execute('SELECT * FROM evaluations ORDER BY pair_index ASC')
#         rows = cursor.fetchall()
#         column_names = [description[0] for description in cursor.description]

#     si = StringIO()
#     cw = csv.writer(si)
#     cw.writerow(column_names)
#     cw.writerows(rows)
    
#     # 添加BOM头，防止用Excel打开CSV时中文乱码
#     response_data = '\ufeff' + si.getvalue()
    
#     return Response(
#         response_data,
#         mimetype="text/csv",
#         headers={"Content-disposition": "attachment; filename=gsb_results.csv"}
#     )

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)


import os
import random
import sqlite3
from flask import Flask, render_template, request, jsonify, Response, send_file
import csv
from io import StringIO

app = Flask(__name__)

# ============ 新的文件夹配置 ============
OURS_DIR = '/m2v_intern/mengzijie/VBench/testdataset/ours_concat'
COMPETITOR_DIRS = {
    'camclone': '/m2v_intern/mengzijie/VBench/testdataset/camclone_concat',
    'ltx': '/m2v_intern/mengzijie/VBench/testdataset/ltx_concat',
    'sd2.0': '/m2v_intern/mengzijie/VBench/testdataset/sd2.0_concat'
}
DB_PATH = 'eval_results.db'

def init_db():
    """初始化任务表与结果表"""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # 任务表 (用于并发控制)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_filename TEXT,
                competitor_name TEXT,
                video_ours_path TEXT,
                video_comp_path TEXT,
                status TEXT DEFAULT 'pending',
                locked_at DATETIME
            )
        ''')
        # 评价结果表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER,
                video_filename TEXT,
                competitor_name TEXT,
                evaluator_id TEXT,
                camera_motion TEXT,
                image_quality TEXT,
                dynamic_rationality TEXT,
                aesthetics TEXT,
                overall TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

def sync_tasks():
    """每次启动应用时，扫描目录并更新任务池 (只增不减)"""
    if not os.path.exists(OURS_DIR):
        print(f"警告：找不到 ours 目录 {OURS_DIR}")
        return
        
    videos_ours = sorted([f for f in os.listdir(OURS_DIR) if f.endswith('.mp4')])
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        for v_name in videos_ours:
            path_ours = os.path.join(OURS_DIR, v_name)
            
            # 与其他三个模型分别进行配对
            for comp_name, comp_dir in COMPETITOR_DIRS.items():
                path_comp = os.path.join(comp_dir, v_name)
                
                # 如果对手文件夹中存在同名视频，则构建一对比较任务
                if os.path.exists(path_comp):
                    cursor.execute('SELECT id FROM tasks WHERE video_filename=? AND competitor_name=?', (v_name, comp_name))
                    if not cursor.fetchone():
                        cursor.execute('''
                            INSERT INTO tasks (video_filename, competitor_name, video_ours_path, video_comp_path, status)
                            VALUES (?, ?, ?, ?, 'pending')
                        ''', (v_name, comp_name, path_ours, path_comp))
        conn.commit()

# 初始化并同步任务池
init_db()
sync_tasks()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/videos/<path:filepath>')
def serve_video(filepath):
    """单独的路由，用于突破Flask限制，直接提供绝对路径的视频流"""
    # 前端传来的路径是去除了开头的'/'的，这里补回来还原绝对路径
    return send_file('/' + filepath)

@app.route('/api/get_task', methods=['GET'])
def get_task():
    # 使用独占事务来处理高并发锁，确保多人同时点击不会发同一个任务
    with sqlite3.connect(DB_PATH, timeout=15, isolation_level='EXCLUSIVE') as conn:
        cursor = conn.cursor()
        cursor.execute('BEGIN EXCLUSIVE')
        
        # 1. 释放锁定时限超过 30 分钟的“僵尸任务”（比如有人点开后关了网页）
        cursor.execute('''
            UPDATE tasks 
            SET status='pending', locked_at=NULL 
            WHERE status='locked' AND locked_at <= datetime('now', '-30 minutes')
        ''')
        
        # 2. 获取所有待评测任务 (彻底混合了三个模型)
        cursor.execute('SELECT id, video_filename, competitor_name, video_ours_path, video_comp_path FROM tasks WHERE status="pending"')
        pending_tasks = cursor.fetchall()
        
        # 3. 统计全局进度
        cursor.execute('SELECT count(*) FROM tasks WHERE status="completed"')
        completed_count = cursor.fetchone()[0]
        cursor.execute('SELECT count(*) FROM tasks')
        total_count = cursor.fetchone()[0]
        
        if not pending_tasks:
            return jsonify({"error": "🎉 太棒了！所有视频对均已评测完毕！"}), 404
            
        # 4. 随机挑选一个任务
        task = random.choice(pending_tasks)
        task_id, video_filename, competitor_name, video_ours_path, video_comp_path = task
        
        # 5. 给该任务加锁
        cursor.execute('UPDATE tasks SET status="locked", locked_at=datetime("now") WHERE id=?', (task_id,))
        conn.commit()
        
    # 随机打乱左右顺序，实现完全盲测
    is_ours_left = random.choice([True, False])
    
    if is_ours_left:
        left_url = f"videos/{video_ours_path.lstrip('/')}"
        right_url = f"videos/{video_comp_path.lstrip('/')}"
    else:
        left_url = f"videos/{video_comp_path.lstrip('/')}"
        right_url = f"videos/{video_ours_path.lstrip('/')}"
        
    return jsonify({
        "task_id": task_id,
        "video_filename": video_filename,
        "competitor_name": competitor_name,
        "left_url": left_url,
        "right_url": right_url,
        "is_ours_left": is_ours_left,
        "completed_count": completed_count,
        "total_count": total_count
    })

@app.route('/api/submit', methods=['POST'])
def submit():
    data = request.json
    task_id = data.get('task_id')
    video_filename = data.get('video_filename')
    competitor_name = data.get('competitor_name')
    is_ours_left = data.get('is_ours_left')
    metrics = data.get('metrics')
    
    # 将前端的 Left/Right/Tie 映射回真实模型名字 (Ours / Competitor / Tie)
    def map_result(choice):
        if choice == 'Tie': return 'Tie'
        if choice == 'Left': return 'Ours' if is_ours_left else competitor_name
        if choice == 'Right': return competitor_name if is_ours_left else 'Ours'
        return 'Unknown'
    
    mapped_metrics = {k: map_result(v) for k, v in metrics.items()}
    
    with sqlite3.connect(DB_PATH, timeout=15) as conn:
        cursor = conn.cursor()
        cursor.execute('BEGIN EXCLUSIVE')
        
        # 将评测结果写入后台
        cursor.execute('''
            INSERT INTO evaluations 
            (task_id, video_filename, competitor_name, evaluator_id, camera_motion, image_quality, dynamic_rationality, aesthetics, overall)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            task_id,
            video_filename,
            competitor_name,
            request.remote_addr, 
            mapped_metrics.get('camera_motion'),
            mapped_metrics.get('image_quality'),
            mapped_metrics.get('dynamic_rationality'),
            mapped_metrics.get('aesthetics'),
            mapped_metrics.get('overall')
        ))
        
        # 解锁任务并标记为已完成
        cursor.execute('UPDATE tasks SET status="completed" WHERE id=?', (task_id,))
        conn.commit()
        
    return jsonify({"status": "success"})

@app.route('/export')
def export_csv():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM evaluations ORDER BY id ASC')
        rows = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]

    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(column_names)
    cw.writerows(rows)
    
    response_data = '\ufeff' + si.getvalue()
    
    return Response(
        response_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=gsb_results.csv"}
    )

if __name__ == '__main__':
    # 开启 threaded=True 支持并发请求
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)