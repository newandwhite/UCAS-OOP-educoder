### 第1关：什么是类，如何创建类

```
package step1;

public class Test {
    public static void main(String[] args) {
        /********** Begin **********/
        //创建Dog对象
        //设置Dog对象的属性
        Dog wuhuarou = new Dog();
        wuhuarou.name = "五花肉";
        wuhuarou.color = "棕色";
        wuhuarou.variety = "阿拉斯加";

        //输出小狗的属性
        System.out.println("名字：" + wuhuarou.name + "，毛色：" + wuhuarou.color + "，品种：" +  wuhuarou.variety);

        //调用方法
        wuhuarou.eat();
        wuhuarou.run();

        /********** End **********/

    }
}

//在这里定义Dog类
/********** Begin **********/
class Dog{
    String name;
    String color;
    String variety;
    void eat(){
        System.out.println("啃骨头");
    }
    void run(){
        System.out.println("叼着骨头跑");
    }
}


/********** End **********/
```

### 第2关：构造方法

```
package step2;

import java.util.Scanner;

public class Test {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        String name = sc.next();
        String sex = sc.next();
        /********** Begin **********/
        //分别使用两种构造器来创建Person对象  
        Person stu = new Person();
        Person stu1 = new Person(name, sex);

        /********** End **********/

    }
}

//创建Person对象，并创建两种构造方法
/********** Begin **********/
class Person{
    public Person(){
        System.out.println("一个人被创建了");
    }
    public Person(String name, String sex){
        System.out.println("姓名：" + name + "，性别：" + sex + "，被创建了");
    }
}


/********** End **********/
```

### 第3关：选择题\(一\)

1. C
2. CD

### 第4关：This关键字

```
package step3;

import java.util.Scanner;

public class Test {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        String name = sc.next();
        int age = sc.nextInt();
        String sex = sc.next();
        Person p = new Person(name,age,sex);
        p.display();
    }
}

class Person{
    String name = "张三";
    int age = 18; 
    String sex = "男";
    /********** Begin **********/

    public Person(String name,int age,String sex){
        this.name = name;
        this.sex = sex;
        this.age = age;
    }

    public void display(){
        // String name = "李四";
        // int age = 11;
        // String sex = "男";
        System.out.println("name：" + name);
        System.out.println("age：" + age);
        System.out.println("sex：" + sex);
    }


    /********** End **********/
}
```

### 第5关：类与对象练习

```
package step4;

import java.util.Scanner;

public class Test {

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        String theMa = sc.next();
        int quantity = sc.nextInt();
        boolean likeSoup = sc.nextBoolean();
        /********** Begin **********/
        //使用三个参数的构造方法创建WuMingFen对象  取名 f1
        WuMingFen f1=new WuMingFen(theMa,quantity,likeSoup);
        //使用两个参数的构造方法创建WuMingFen对象  取名 f2
        WuMingFen f2=new WuMingFen(theMa,quantity);
        //使用无参构造方法创建WuMingFen对象  取名 f3
        WuMingFen f3=new WuMingFen();
        //分别调用三个类的 check方法
        f1.check();
        f2.check();
        f3.check();
        /********** End **********/    
    }
}
```

### 第6关：static关键字

```
package step5;

public class Test {
	/********** Begin **********/
	static String name = "楚留香";
	
	static{
		System.out.println("hello educoder");
	}
	public static void main(String[] args) {
		System.out.println("我叫" + name);
		study();
	}
	
	public static void study(){
		System.out.println("我喜欢在educoder上学习java");
	}
	/********** End **********/
}

```

### 第7关：选择题\(二\)

1. D
2. EG
3. B



