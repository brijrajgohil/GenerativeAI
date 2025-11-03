from pydantic import BaseModel, Field
from typing import Optional
class Student(BaseModel):
    name: str = 'john'
    age: Optional[int] = None
    cgpa: float = Field(gt=0, lt=10, default=5.0, description="Decimal value representing CGPA of a student")

new_student = {"name": "John Doe", "age": "25", "cgpa": 8.5}
student = Student(**new_student)
print(student)
print(type(student))