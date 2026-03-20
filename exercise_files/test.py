# Python code​​​​​​‌‌‌‌​‌​‌‌​‌​​​‌​‌‌‌‌‌‌‌​​ below

def factorial(num):
    # Your code goes here.
    if num==0:
        result=1
        if num>0:
            result=num
            while num>0:
                result=result*num-1
                num=num-1
            else:
                print("negative number not allowed")
        
    return num  
    

print(factorial(5))
