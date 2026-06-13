import sqlite3
import os

# 指定你的数据库绝对路径（防止相对路径找错）todo
DB_PATH = '/m2v_intern/mengzijie/m2v_camclone_v2/gsb_eval/eval_results.db'

def undo_last_eval():
    if not os.path.exists(DB_PATH):
        print(f"❌ 找不到数据库文件: {DB_PATH}")
        return

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        # 1. 查找最后一次的评测记录
        cursor.execute("SELECT id, task_id, video_filename, competitor_name FROM evaluations ORDER BY id DESC LIMIT 1")
        last_record = cursor.fetchone()
        
        if not last_record:
            print("⚠️ 当前数据库中没有任何评测记录，无需撤销。")
            return
            
        eval_id, task_id, video_filename, comp_name = last_record
        print(f"🔍 找到最后一条记录:")
        print(f"   - 视频名: {video_filename}")
        print(f"   - 对比模型: {comp_name}")
        
        # 2. 确认撤销：从 evaluations 表删除该记录
        cursor.execute("DELETE FROM evaluations WHERE id=?", (eval_id,))
        
        # 3. 释放任务池中的该任务：状态改回 pending
        cursor.execute("UPDATE tasks SET status='pending', locked_at=NULL WHERE id=?", (task_id,))
        conn.commit()
        
        print(f"\n✅ 成功撤销！视频 [{video_filename}] 已被重新放回盲测任务池，下次会重新刷出。")

if __name__ == "__main__":
    undo_last_eval()