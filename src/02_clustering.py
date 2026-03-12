import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
from pathlib import Path

INPUT_PATH = Path('data/processed_base.csv')
OUTPUT_PATH = Path('data/clustered_students.csv')
IMG_DIR = Path('img')

def main():
    df = pd.read_csv(INPUT_PATH)
    IMG_DIR.mkdir(exist_ok=True)

    features = ['Score_Level', 'Num_Tutor_Subjects', 'Time_Lag', 'Enrolled_Time']
    X = df[features]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    results = []
    range_n_clusters = range(2, 11)

    print("Поиск оптимального числа кластеров...")
    for n in range_n_clusters:
        clusterer = AgglomerativeClustering(n_clusters=n, linkage='ward')
        cluster_labels = clusterer.fit_predict(X_scaled)
        
        sil_score = silhouette_score(X_scaled, cluster_labels)
        db_score = davies_bouldin_score(X_scaled, cluster_labels)
        
        results.append({'K': n, 'Silhouette': sil_score, 'Davies_Bouldin': db_score})
        print(f"K={n:2} | Silhouette: {sil_score:.3f} | Davies-Bouldin: {db_score:.3f}")

    results_df = pd.DataFrame(results)
    best_k = results_df.loc[results_df['Silhouette'].idxmax(), 'K']
    print(f"\nЛучший результат по Silhouette при K={int(best_k)}")

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.plot(results_df['K'], results_df['Silhouette'], 'g-o', label='Silhouette (Higher is better)')
    ax2.plot(results_df['K'], results_df['Davies_Bouldin'], 'b-o', label='Davies-Bouldin (Lower is better)')
    ax1.set_xlabel('Количество кластеров')
    ax1.set_ylabel('Silhouette Score', color='g')
    ax2.set_ylabel('Davies-Bouldin Score', color='b')
    plt.title('Выбор K: Silhouette vs Davies-Bouldin')
    plt.savefig(IMG_DIR / 'metrics.png', bbox_inches='tight')
    plt.close()

    model = AgglomerativeClustering(n_clusters=int(best_k), linkage='ward')
    df['Cluster'] = model.fit_predict(X_scaled)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2d = tsne.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=df['Cluster'], palette='tab10', s=60, alpha=0.8)
    plt.title(f'Визуализация кластеров (t-SNE проекция, K={int(best_k)})')
    plt.legend(title='Кластер', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(IMG_DIR / 'tsne_clusters.png', bbox_inches='tight')
    plt.close()

    print("\n" + "="*40)
    print(f"ОПИСАНИЕ КЛАСТЕРОВ (K={int(best_k)})")
    print("="*40)

    cluster_profiles = df.groupby('Cluster').agg(
        Size=('School_Type', 'count'),
        Lyceum_Share=('School_Type', 'mean'), 
        Score_Mean=('Score_Level', 'mean'),
        Tutors_Mean=('Num_Tutor_Subjects', 'mean'),
        Lag_Mean=('Time_Lag', 'mean'),
        Start_Mean=('Enrolled_Time', 'mean')
    ).sort_values(by='Size', ascending=False)

    for cluster_id, row in cluster_profiles.iterrows():
        print(f"\nКластер {cluster_id} | Учеников: {int(row['Size'])}")
        print(f"  Доля лицеистов: {row['Lyceum_Share'] * 100:>5.1f}%")
        print(f"  Амбиции (1-4):  {row['Score_Mean']:>5.2f}")
        print(f"  Репетиторов:    {row['Tutors_Mean']:>5.2f}")
        print(f"  Time Lag (0-6): {row['Lag_Mean']:>5.2f}")
        print(f"  Время старта:   {row['Start_Mean']:>5.2f}")

    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"\nДанные сохранены в {OUTPUT_PATH}")
    print(f"Графики сохранены в директорию {IMG_DIR}/")

if __name__ == "__main__":
    main()