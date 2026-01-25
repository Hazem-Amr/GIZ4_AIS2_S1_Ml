from abc import ABC, abstractmethod

class Person(ABC):
    def __init__(self, name , age ):
        self.name = name
        self.age = age

    @abstractmethod
    def view_info(self):
        pass


class Patient(Person):

    def __init__(self, name , age , medical_record):
        super().__init__(name , age)
        self.medical_record = medical_record

    def view_info(self):
        return f"Name is {self.name} , Age is {self.age}"
    
    def view_record(self):
        return f"The medical record is {self.medical_record}"
    


class Staff(Person):

    def __init__(self, name , age , position):
        super().__init__(name , age)
        self.position = position

    def view_info(self):
        return f"Name is {self.name} , Age is {self.age} , Position is {self.position}"
    


class Department:

    def __init__(self , name):
        self.name = name
        self.patients = []
        self.staff = []

    def add_patient(self, patient):
        if isinstance(patient , Patient):
            self.patients.append(patient)
            print("Patient Added Successfully")
            return patient
        else: 
            print("Something Went Wrong")
            print(f"Check for the current object status and try again\nobject: {patient}")
            return patient
        
    def add_staff(self, staff):
        if isinstance(staff , Staff):
            self.staff.append(staff)
            print("Staff Added Successfully")
            return staff
        else: 
            print("Something Went Wrong")
            print(f"Check for the current object status and try again\nobject: {staff}")
            return staff


class Hospital:
    def __init__(self, name, location):
        self.name = name
        self.location = location
        self.departments = []

    def add_department(self , department):

        if isinstance(department , Department):

            self.departments.append(department)
            print("Department Added Successfully")
            return department
        
        else:
            print("Something Went Wrong")
            print(f"Check for the current object status and try again\nobject: {department}")
            return department
