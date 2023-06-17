@echo off
call "C:\Users\acer\anaconda3\Scripts\activate.bat" myenv
cd /d "E:\Stock\Bot"
python Analyse.py %1
