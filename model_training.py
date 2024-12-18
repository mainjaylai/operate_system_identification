import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_data(train_file, test_file):
    """加载训练集和测试集"""
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    return train_df, test_df


def preprocess_data(train_df, test_df):
    """数据预处理"""
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    return X_train, y_train, X_test, y_test


def plot_metrics(history, y_test, y_pred, voting_clf, X_test):
    """绘制模型评估指标"""
    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['test_loss'], label='测试损失')
    plt.title('损失曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    plt.legend()
    plt.savefig('loss_curve.png')  # 保存损失曲线
    plt.show()

    # 计算并绘制其他指标
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)

    metrics = {'准确率': accuracy, '精确率': precision, '召回率': recall, 'F1值': f1}
    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics.keys(), metrics.values())
    plt.title('模型评估指标')
    plt.ylabel('值')
    
    # 在每个条形上方显示数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom')
    
    plt.savefig('metrics.png')  # 保存模型评估指标
    plt.show()

    # 绘制混淆矩阵
    cm_display = ConfusionMatrixDisplay.from_estimator(voting_clf, X_test, y_test)
    cm_display.ax_.set_title('混淆矩阵')
    plt.savefig('confusion_matrix.png')  # 保存混淆矩阵
    plt.show()


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """训练模型并评估"""
    # 定义各个模型
    knn = KNeighborsClassifier(n_neighbors=5)
    svm = SVC(probability=True, decision_function_shape="ovr")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(max_iter=1000)

    # 创建投票分类器
    voting_clf = VotingClassifier(
        estimators=[("knn", knn), ("svm", svm), ("rf", rf), ("lr", lr)], voting="soft"
    )

    # 训练模型
    voting_clf.fit(X_train, y_train)
    # 评估模型
    y_pred = voting_clf.predict(X_test.to_numpy())
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n使用选定特征的集成模型准确率: {accuracy:.2f}")

    # 打印分类报告和混淆矩阵
    print("分类报告:")
    print(classification_report(y_test, y_pred))
    print("混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))

    # 保存模型
    joblib.dump(voting_clf, "voting_classifier.pkl")
    print("模型已保存为 voting_classifier.pkl")

    # 绘制评估指标
    plot_metrics({'train_loss': [], 'test_loss': []}, y_test, y_pred, voting_clf, X_test)


def main():
    train_file = "combined_train.csv"
    test_file = "combined_test.csv"

    train_df, test_df = load_data(train_file, test_file)
    X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df)
    train_and_evaluate(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()

