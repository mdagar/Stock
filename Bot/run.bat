@echo off
call "C:\Users\acer\anaconda3\Scripts\activate.bat" myenv
cd /d "E:\Stock\Bot"
python bot.py %1
