from factory import Staff , Patient , Hospital , Department



h = Hospital("bolaq" , 'giza')

d = Department("batna")

s = Staff('hazem' , 22 , "doctor")
p = Patient("ahmed" , 22 , "cold")


d.add_patient(p)
d.add_staff(s)


h.add_department(d)


print(h.departments)
