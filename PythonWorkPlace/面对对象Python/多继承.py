class Base:
    def fight(self):
        print("动物在打架。。。")


class ShenXianBase(Base):
    def fight(self):
        print("始祖在打架。。。")


class ShenXian(ShenXianBase):
    def fly(self):
        print("在飞。。。")

    def fight(self):
        print("神仙打架。。。")


class MonkeyBase(Base):
    def fight(self):
        print("猿猴在打架。。。")


class Monkey(MonkeyBase):
    def eat_peach(self):
        print("在吃。。。")

    def fight(self):
        print("猴子打架。。。")


class MonkeyKing(ShenXian, Monkey):
    def play_strick(self):
        print("打架。。。")


m = MonkeyKing()
m.fly()
m.play_strick()
m.eat_peach()
m.fight()  # 关于这个 def 是继承了谁的方法，这个是根据 深度优先 查找方法进行的。
