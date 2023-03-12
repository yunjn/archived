create table 教师信息表
(职工号 int not null primary key, 
姓名 varchar(10), 
性别 char check( 性别='男'or 性别='女'), 
电话 varchar(15), 
科目 varchar(15), 
地址 varchar(20));

create table 职业登记表
( 职业号 varchar (20) primary key,
名称 varchar (20),
电话 varchar (20),
地址 varchar (20));

create table 职工作息表
( 职工号 int, 
开始时间 varchar (15), 
结束时间 varchar (15), 
工作时间 int primary key,
日期 varchar(15));

create table 收费表
( 职工号 int , 
开始时间 varchar (15), 
结束时间 varchar (15), 
收费 int primary key);

create table 工资表
( 职工号 int not null primary key,
工资 float);