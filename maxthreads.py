import threading
import time 

def mythread():
    time.sleep(1000)

def main():
    threads = 0
    y=  1000000
    for i in range(y):
        try:
            x=threading.Thread(target=mythread, daemon=True)
            threads +=1
            x.start()
        except RuntimeError:
            break
    print(threads)

main()
