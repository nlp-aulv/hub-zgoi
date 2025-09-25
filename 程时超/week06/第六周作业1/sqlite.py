import sqlite3

# 连接到学校数据库，如果文件不存在会自动创建
conn = sqlite3.connect('School.db')
cursor = conn.cursor()

# 创建 students 表
cursor.execute('''
CREATE TABLE IF NOT EXISTS students (
    student_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER,
    major TEXT
);
''')

# 创建 courses 表
cursor.execute('''
CREATE TABLE IF NOT EXISTS courses (
    course_id INTEGER PRIMARY KEY,
    course_name TEXT NOT NULL,
    instructor TEXT,
    credits INTEGER
);
''')

# 创建 grades 表，外键关联 students 和 courses 表
cursor.execute('''
CREATE TABLE IF NOT EXISTS grades (
    grade_id INTEGER PRIMARY KEY,
    student_id INTEGER,
    course_id INTEGER,
    grade REAL,
    semester TEXT,
    FOREIGN KEY (student_id) REFERENCES students (student_id),
    FOREIGN KEY (course_id) REFERENCES courses (course_id)
);
''')

# 提交更改
conn.commit()
print("学校数据库和表已成功创建。")

# 插入学生数据
cursor.execute("INSERT INTO students (name, age, major) VALUES (?, ?, ?)", 
               ('张三', 18, '科学'))
cursor.execute("INSERT INTO students (name, age, major) VALUES (?, ?, ?)", 
               ('李四', 20, '人文'))
cursor.execute("INSERT INTO students (name, age, major) VALUES (?, ?, ?)", 
               ('王五', 19, '艺术'))
conn.commit()

# 插入课程数据
cursor.execute("INSERT INTO courses (course_name, instructor, credits) VALUES (?, ?, ?)", 
               ('数学', '张老师', 4))
cursor.execute("INSERT INTO courses (course_name, instructor, credits) VALUES (?, ?, ?)", 
               ('语文', '李老师', 3))
cursor.execute("INSERT INTO courses (course_name, instructor, credits) VALUES (?, ?, ?)", 
               ('物理', '王教授', 4))
conn.commit()

# 插入成绩数据
# 获取学生ID
cursor.execute("SELECT student_id FROM students WHERE name = '张三'")
student1_id = cursor.fetchone()[0]

cursor.execute("SELECT student_id FROM students WHERE name = '李四'")
student2_id = cursor.fetchone()[0]

# 获取课程ID
cursor.execute("SELECT course_id FROM courses WHERE course_name = '数学'")
math_id = cursor.fetchone()[0]

cursor.execute("SELECT course_id FROM courses WHERE course_name = '语文'")
chinese_id = cursor.fetchone()[0]

# 插入成绩记录
cursor.execute("INSERT INTO grades (student_id, course_id, grade, semester) VALUES (?, ?, ?, ?)",
               (student1_id, math_id, 95.5, '2025夏季'))
cursor.execute("INSERT INTO grades (student_id, course_id, grade, semester) VALUES (?, ?, ?, ?)",
               (student1_id, chinese_id, 88.0, '2025夏季'))
cursor.execute("INSERT INTO grades (student_id, course_id, grade, semester) VALUES (?, ?, ?, ?)",
               (student2_id, math_id, 92.0, '2025夏季'))
conn.commit()

print("学校数据已成功插入。")

# 查询所有学生及其成绩
print("\n--- 学生成绩报告 ---")
cursor.execute('''
SELECT students.name, courses.course_name, grades.grade, grades.semester
FROM grades
JOIN students ON grades.student_id = students.student_id
JOIN courses ON grades.course_id = courses.course_id
ORDER BY students.name;
''')

grade_report = cursor.fetchall()
for name, course, grade, semester in grade_report:
    print(f"学生: {name}, 课程: {course}, 成绩: {grade}, 学期: {semester}")

# 更新学生信息
print("\n--- 更新学生信息 ---")
cursor.execute("UPDATE students SET age = ? WHERE name = ?", (19, '张三'))
conn.commit()
print("学生 '张三' 的年龄已更新。")

# 查询更新后的学生信息
cursor.execute("SELECT name, age, major FROM students WHERE name = '张三'")
updated_student = cursor.fetchone()
print(f"更新后的信息: 姓名: {updated_student[0]}, 年龄: {updated_student[1]}, 专业: {updated_student[2]}")

# 删除一个课程
print("\n--- 删除课程 ---")
cursor.execute("DELETE FROM courses WHERE course_name = ?", ('物理',))
conn.commit()
print("课程 '物理' 已被删除。")

# 再次查询课程列表，验证删除操作
print("\n--- 剩余的课程 ---")
cursor.execute("SELECT course_name, instructor FROM courses")
remaining_courses = cursor.fetchall()
for course in remaining_courses:
    print(f"课程: {course[0]}, 教师: {course[1]}")

# 关闭连接
conn.close()
print("\n数据库连接已关闭。")
