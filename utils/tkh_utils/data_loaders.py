def load_titanic():
    """Load and return cleaned Titanic dataset.
    Returns: DataFrame"""
    import seaborn as sns
    df = sns.load_dataset('titanic')
    return df


def load_heart_disease():
    """Load Heart Disease dataset.
    Returns: X (DataFrame), y (Series)"""
    from sklearn.datasets import fetch_openml
    heart = fetch_openml(name='heart-disease', version=1, as_frame=True)
    X = heart.data.drop('target', axis=1)
    y = heart.data['target'].astype(int)
    return X, y


def load_california_housing():
    """Load California Housing dataset.
    Returns: X (DataFrame), y (Series)"""
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing(as_frame=True)
    return housing.data, housing.target


def load_airbnb():
    """Load NYC Airbnb dataset.
    Loads from local file if available, falls back to URL in Codespaces.
    Returns: DataFrame"""
    import os
    import pandas as pd

    local_path = os.path.join(
        os.path.dirname(__file__),
        '../../data/nyc_airbnb.csv'
    )

    if os.path.exists(local_path):
        return pd.read_csv(local_path)

    print("Loading Airbnb dataset from URL...")
    url = ("https://raw.githubusercontent.com/"
           "dsrscientist/dataset1/master/"
           "new-york-city-airbnb-open-data.csv")
    df = pd.read_csv(url)

    data_dir = os.path.join(
        os.path.dirname(__file__),
        '../../data'
    )
    if os.path.exists(data_dir):
        df.to_csv(local_path, index=False)
        print(f"Cached to {local_path}")

    return df


def load_diabetes():
    """Load Pima Indians Diabetes dataset.
    Returns: X (DataFrame), y (Series)"""
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name='diabetes', version=1, as_frame=True)
    return data.data, data.target.astype(int)


def load_student_performance():
    """Load Student Performance dataset.
    Returns: X (DataFrame), y (Series)"""
    from sklearn.datasets import fetch_openml
    data = fetch_openml(
        name='student-performance-por',
        version=1, as_frame=True
    )
    return data.data, data.target


def load_bank_marketing():
    """Load Bank Marketing Campaign dataset.
    Returns: X (DataFrame), y (Series)"""
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name='bank-marketing', version=1, as_frame=True)
    return data.data, data.target


def load_mall_customers():
    """Load Mall Customers dataset.
    Loads from local file if available, falls back to URL in Codespaces.
    Returns: DataFrame"""
    import os
    import pandas as pd

    local_path = os.path.join(
        os.path.dirname(__file__),
        '../../data/mall_customers.csv'
    )

    if os.path.exists(local_path):
        return pd.read_csv(local_path)

    print("Loading Mall Customers from URL...")
    url = ("https://raw.githubusercontent.com/"
           "dsrscientist/dataset1/master/"
           "Mall_Customers.csv")
    df = pd.read_csv(url)

    data_dir = os.path.join(
        os.path.dirname(__file__),
        '../../data'
    )
    if os.path.exists(data_dir):
        df.to_csv(local_path, index=False)
        print(f"Cached to {local_path}")

    return df


def load_world_happiness():
    """Load World Happiness Report dataset.
    Loads from local file if available, falls back to URL in Codespaces.
    Returns: DataFrame"""
    import os
    import pandas as pd

    local_path = os.path.join(
        os.path.dirname(__file__),
        '../../data/world_happiness.csv'
    )

    if os.path.exists(local_path):
        return pd.read_csv(local_path)

    print("Loading World Happiness from URL...")
    url = ("https://raw.githubusercontent.com/"
           "dsrscientist/dataset1/master/"
           "world-happiness-report.csv")
    df = pd.read_csv(url)

    data_dir = os.path.join(
        os.path.dirname(__file__),
        '../../data'
    )
    if os.path.exists(data_dir):
        df.to_csv(local_path, index=False)
        print(f"Cached to {local_path}")

    return df
