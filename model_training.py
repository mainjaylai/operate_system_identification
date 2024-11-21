import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


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


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """训练模型并评估"""
    # 定义各个模型
    knn = KNeighborsClassifier(n_neighbors=5)
    svm = SVC(probability=True, decision_function_shape='ovr')
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(max_iter=1000)

    # 创建投票分类器
    voting_clf = VotingClassifier(
        estimators=[("knn", knn), ("svm", svm), ("rf", rf), ("lr", lr)], voting="soft"
    )

    # 训练模型
    voting_clf.fit(X_train, y_train)

    # 评估模型
    y_pred = voting_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"集成模型准确率: {accuracy:.2f}")

    # 打印分类报告和混淆矩阵
    print("分类报告:")
    print(classification_report(y_test, y_pred))
    print("混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))

    # 保存模型
    joblib.dump(voting_clf, "voting_classifier.pkl")
    print("模型已保存为 voting_classifier.pkl")


def main():
    train_file = "feature_files/combined_train.csv"
    test_file = "feature_files/combined_test.csv"

    train_df, test_df = load_data(train_file, test_file)
    X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df)
    train_and_evaluate(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
