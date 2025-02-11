
"""

Investment Management and Machine Learning

Week 1

Instructor:Wei Jiao

"""


print ("Hello World!")

# Select lines of code and then press F9 to run the selected code

print ("Hello World!")



""" Comments

Comments are the useful information that the developers provide to make the developers and readers
understand the source code. 

Comments will not be executed

1. Single line comment starts with #  hashtag symbol 
2. Multi-line comment should be enclosed in two delimiters (three quotation marks)

""" 

"""

"""



#This is a comment
print ("Hello World!")



"""
This is a comment.

Comments will not be executed.

"""
print ("Hello World!")





"""Declare variables """

myNumber = 3

print(myNumber) 


Text1 ="Hello World"

#variable type is String
print(Text1) 


#Assign multiple variables

a,b,c=1,2,3




"""List""" 

list1 = [1, 2, 3, 4] 

list1=[1,2,3,4]


list1[0] #select the first element from the list. 0 refers to the first element


first_num=list1[0]


list1[3] # select the fourth element


list1[0:3]  
# pick multiple elements from the list
#[start:stop]. not including the stop element
#from the first element to the third element
#3 indicates the fourth element but this selection does not include the fourth element


list1[1:] # select from the second element to the last element

list1[:3] # select from the first element to the third element



# creates a empty list 
nums = [] 

# append data in list 
nums.append(1) 
nums.append(2) 
nums.append(3) 

print(nums) 




"""Math Calculation"""

1+2

2*3

5/2

2**2

2**3


"""
Python Arithmetic Operators
https://www.geeksforgeeks.org/python-arithmetic-operators/
"""



""" Define functions """

def hello(): 
	print("Hello World") 

    
# calling functions
hello() 


# define summation function
def mysum(x,y):
    return x+y  

"""We need an indented block before return x+y.
This indented block indicates return x+y is part of the function
"""

mysum(1,2)

def afunction(a,b,c):
    return a+b-c

afunction(5,5,5)


#Wrong one!!
def mysum(x,y):
return x+y

"""Use TAB or Space key to generate the indented block"""


#Another function
def fun1(x,y,z):
    return x**2+y**(1/2)+z*x 

fun1(1,2,3)

fun1(10,20,100)



"""loop

a loop is a programming structure that repeats a sequence of instructions 

until a specific condition is met.

""" 


#print all the integers from 0 to 4
for i in range(5):	
    print(i) #We need an indented block 


for i in range(0,5):	
    print(i) 
#range(start, stop)


for i in range(0,5,2):	
    print(i) 
#range(start, stop,step)  
#step: integer value which determines the increment between each integer in the sequence 
    
for i in range(5,0,-1):	
    print(i) 


#generate a list of numbers
nums = [] 

for i in range(1,5):
    nums.append(i)

print (nums)


#sum all the numbers from 1 to 4
sum1=0

for i in range(1,5):
    sum1=sum1+i
    
print (sum1)

"""
Python For Loop
https://www.geeksforgeeks.org/python-for-loops/
"""











