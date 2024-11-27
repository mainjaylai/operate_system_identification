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
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 首先训练随机森林来分析特征重要性
    rf.fit(X_train, y_train)
    
    # 获取特征重要性
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf.feature_importances_
    })
    
    # 按重要性排序并显示
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print("\n特征重要性排序:")
    print(feature_importance)
    
    # 只选择重要性大于平均值的特征
    important_features = feature_importance[
        feature_importance['importance'] > feature_importance['importance'].mean()
    ]['feature'].tolist()
    
    print(f"\n建议使用的重要特征 (重要性高于平均值):")
    print(important_features)
    
    # 使用筛选后的特征进行训练
    X_train_selected = X_train[important_features]
    X_test_selected = X_test[important_features]
    
    # 定义其他模型
    knn = KNeighborsClassifier(n_neighbors=5)
    svm = SVC(probability=True, decision_function_shape="ovr")
    lr = LogisticRegression(max_iter=1000)

    # 创建投票分类器
    voting_clf = VotingClassifier(
        estimators=[("knn", knn), ("svm", svm), ("rf", rf), ("lr", lr)], 
        voting="soft"
    )

    # 使用选定的特征训练模型
    voting_clf.fit(X_train_selected, y_train)
    
    # 评估模型
    y_pred = voting_clf.predict(X_test_selected)
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


def main():
    train_file = "combined_train.csv"
    test_file = "combined_test.csv"

    train_df, test_df = load_data(train_file, test_file)
    X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df)
    train_and_evaluate(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
