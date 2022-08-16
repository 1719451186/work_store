
class Person:
    def __init__(self, name, age, sex, relation):
        self.name = name
        self.age = age
        self.sex = sex
        self.relation = relation
        # None应该是一个对象，代表另一半
        # self.parter = None

    def do_private_stuff(self):
        pass

class RelationShip:
    """保存 couple 之间的对象关系"""
    def __init__(self):
        self.couple = []

    def make_couple(self, obj1, obj2):
        self.couple = [obj1, obj2]
        print("[%s] 和 [%s] 确定了男女关系" % (obj1.name, obj2.name))

    def get_my_parter(self, obj1):
        print("找[%s]的对象" % obj1.name)
        for i in self.couple:
            if i != obj1:
                return i
        else:
            print("[%s]是单身" %obj1.name)

    def break_up(self):
        print("[%s]和[%s]分手了" %(self.couple[0].name, self.couple[1].name) )
        self.couple.clear()



relation_obj = RelationShip()
p1 = Person("wboy", 24, "M", relation_obj)
p2 = Person("NvPen", 23, "F", relation_obj)
relation_obj.make_couple(p1, p2)

print(p1.relation.couple)

p1.relation.get_my_parter(p1)
print(p1.relation.get_my_parter(p1).name)

p1.relation.break_up()

p2.relation.get_my_parter(p2)