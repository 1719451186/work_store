attack_vals = {
    "京巴": 30,
    "藏獒": 80
}


def dog(name, d_type):  # 模板
    data = {
        "name": name,
        "d_type": d_type,
        "life_val": 100
    }
    if d_type in attack_vals:
        data["attack_val"] = attack_vals[d_type]
    else:
        data["attack_val"] = 15

    def dog_bite(person_obj):
        person_obj["life_val"] -= data["attack_val"]
        print("狗[%s]咬了[%s]一口，[%s]收到了[%s]点伤害，还有血量[%s].."
              % (data["name"],
                 person_obj["name"],
                 person_obj["name"],
                 data["attack_val"],
                 person_obj["life_val"]))

    data["bite"] = dog_bite # 为了从函数外部可以调用 dog_bite 方法。

    return data


def person(name, age):
    data = {
        "name": name,
        "age": age,
        "life_val": 100
    }
    if age > 18:
        data["attack_val"] = 50
    else:
        data["attack_val"] = 30
    return data


def dog_bite(dog_obj, person_obj):
    person_obj["life_val"] -= dog_obj["attack_val"]
    print("狗[%s]咬了[%s]一口，[%s]收到了[%s]点伤害，还有血量[%s].."
          % (dog_obj["name"],
             person_obj["name"],
             person_obj["name"],
             dog_obj["attack_val"],
             person_obj["life_val"]))


def beat(person_obj, dog_obj):
    dog_obj["life_val"] -= person_obj["attack_val"]
    print("人[%s]打了狗[%s]一棒，狗掉了[%s]点血，还有[%s]血"
          % (person_obj["name"],
             dog_obj["name"],
             person_obj["attack_val"],
             dog_obj["life_val"]))


# 实体1
d1 = dog("mjj", "京巴")
# 实体2
d2 = dog("mjj2", "藏獒")
# 实体3
p1 = person("wboy", 23)

# 动作1
d1["bite"](p1)

# 动作2
beat(p1, d1)

# 错误实例
beat(d1, p1)

