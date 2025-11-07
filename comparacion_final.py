# Comparación Final de Todos los Modelos
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configurar estilo
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def load_experimento1_results():
    """Cargar resultados del Experimento 1 (si existen)"""
    results = {}
    
    if os.path.exists("experimento1_resultados.csv"):
        df = pd.read_csv("experimento1_resultados.csv")
        for _, row in df.iterrows():
            results[f"Exp1 - {row['Modelo']}"] = {
                "accuracy": row["Accuracy"],
                "precision": row["Precision"],
                "recall": row["Recall"],
                "f1": row["F1-Score"],
            }
    
    return results


def load_experimento2_results():
    """Cargar resultados del Experimento 2"""
    results = {}
    
    if os.path.exists("experimento2_resultados.csv"):
        df = pd.read_csv("experimento2_resultados.csv")
        for _, row in df.iterrows():
            results[row["Modelo"]] = {
                "accuracy": row["Accuracy"],
                "precision": row["Precision"],
                "recall": row["Recall"],
                "f1": row["F1-Score"],
            }
    
    return results


def create_consolidated_results():
    """
    Crear resultados consolidados leyendo archivos CSV o usando valores por defecto
    Nota: Este script asume que los experimentos ya fueron ejecutados
    """
    all_results = {}
    
    # Cargar resultados del Experimento 1
    exp1_results = load_experimento1_results()
    if exp1_results:
        all_results.update(exp1_results)
        print(f"✅ Cargados {len(exp1_results)} modelos del Experimento 1")
    
    # Cargar resultados del Experimento 2
    exp2_results = load_experimento2_results()
    if exp2_results:
        all_results.update(exp2_results)
        print(f"✅ Cargados {len(exp2_results)} modelos del Experimento 2")
    
    # Si no hay resultados cargados, mostrar mensaje
    if not all_results:
        print("⚠️ No se encontraron archivos de resultados.")
        print("   Ejecuta primero los experimentos:")
        print("   python experimento1.py")
        print("   python experimento2.py")
        return None
    
    return all_results


def create_comparison_table(all_results):
    """Crear tabla comparativa completa"""
    if not all_results:
        return None
    
    comparison_data = []
    for model_name, metrics in all_results.items():
        comparison_data.append({
            "Modelo": model_name,
            "Accuracy": metrics.get("accuracy", 0),
            "Precision": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0),
            "F1-Score": metrics.get("f1", 0),
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values("F1-Score", ascending=False)
    
    return df_comparison


def visualize_comparison(df_comparison):
    """Crear visualizaciones comparativas"""
    if df_comparison is None or len(df_comparison) == 0:
        print("⚠️ No hay datos para visualizar")
        return
    
    # Configurar figura
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Gráfico de barras comparativo (todas las métricas)
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(df_comparison))
    width = 0.2
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    ax1.bar(x - 1.5 * width, df_comparison["Accuracy"], width, label="Accuracy", alpha=0.8, color=colors[0])
    ax1.bar(x - 0.5 * width, df_comparison["Precision"], width, label="Precision", alpha=0.8, color=colors[1])
    ax1.bar(x + 0.5 * width, df_comparison["Recall"], width, label="Recall", alpha=0.8, color=colors[2])
    ax1.bar(x + 1.5 * width, df_comparison["F1-Score"], width, label="F1-Score", alpha=0.8, color=colors[3])
    
    ax1.set_xlabel("Modelos", fontsize=12)
    ax1.set_ylabel("Puntuación", fontsize=12)
    ax1.set_title("Comparación Completa de Modelos - Todas las Métricas", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_comparison["Modelo"], rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim(0, 1)
    
    # 2. Heatmap de métricas
    ax2 = fig.add_subplot(gs[1, 0])
    metrics_matrix = df_comparison.set_index("Modelo")[["Accuracy", "Precision", "Recall", "F1-Score"]]
    sns.heatmap(metrics_matrix, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax2, cbar_kws={"label": "Puntuación"})
    ax2.set_title("Heatmap de Métricas por Modelo", fontsize=12, fontweight="bold")
    ax2.set_ylabel("")
    
    # 3. Gráfico de F1-Score (métrica principal)
    ax3 = fig.add_subplot(gs[1, 1])
    colors_f1 = ['#2ecc71' if x == df_comparison['F1-Score'].max() else '#95a5a6' for x in df_comparison['F1-Score']]
    ax3.barh(range(len(df_comparison)), df_comparison["F1-Score"], color=colors_f1, alpha=0.8)
    ax3.set_yticks(range(len(df_comparison)))
    ax3.set_yticklabels(df_comparison["Modelo"])
    ax3.set_xlabel("F1-Score", fontsize=11)
    ax3.set_title("F1-Score por Modelo (Mejor en verde)", fontsize=12, fontweight="bold")
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis="x")
    ax3.set_xlim(0, 1)
    
    # Agregar valores en las barras
    for i, v in enumerate(df_comparison["F1-Score"]):
        ax3.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=10)
    
    # 4. Radar chart (gráfico de araña) para el mejor modelo
    ax4 = fig.add_subplot(gs[2, 0], projection="polar")
    best_model = df_comparison.iloc[0]
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    metrics_values = [
        best_model["Accuracy"],
        best_model["Precision"],
        best_model["Recall"],
        best_model["F1-Score"],
    ]
    
    # Cerrar el círculo
    angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
    metrics_values += metrics_values[:1]
    angles += angles[:1]
    
    ax4.plot(angles, metrics_values, "o-", linewidth=2, label=best_model["Modelo"], color="#e74c3c")
    ax4.fill(angles, metrics_values, alpha=0.25, color="#e74c3c")
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(metrics_names)
    ax4.set_ylim(0, 1)
    ax4.set_title(f"Perfil del Mejor Modelo: {best_model['Modelo']}", fontsize=12, fontweight="bold", pad=20)
    ax4.grid(True)
    
    # 5. Comparación de Accuracy vs F1-Score
    ax5 = fig.add_subplot(gs[2, 1])
    scatter = ax5.scatter(
        df_comparison["Accuracy"],
        df_comparison["F1-Score"],
        s=200,
        alpha=0.6,
        c=range(len(df_comparison)),
        cmap="viridis",
    )
    
    for idx, row in df_comparison.iterrows():
        ax5.annotate(row["Modelo"], (row["Accuracy"], row["F1-Score"]), fontsize=9, alpha=0.8)
    
    ax5.set_xlabel("Accuracy", fontsize=11)
    ax5.set_ylabel("F1-Score", fontsize=11)
    ax5.set_title("Accuracy vs F1-Score", fontsize=12, fontweight="bold")
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    
    plt.suptitle("Comparación Final de Todos los Modelos", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig("comparacion_final_modelos.png", dpi=300, bbox_inches="tight")
    plt.show()


def generate_summary_report(df_comparison):
    """Generar reporte resumen"""
    if df_comparison is None or len(df_comparison) == 0:
        print("⚠️ No hay datos para generar el reporte")
        return
    
    print("\n" + "=" * 70)
    print("RESUMEN EJECUTIVO - COMPARACIÓN DE MODELOS")
    print("=" * 70)
    
    best_model = df_comparison.iloc[0]
    
    print(f"\n🏆 MEJOR MODELO (según F1-Score): {best_model['Modelo']}")
    print(f"   • F1-Score:    {best_model['F1-Score']:.4f}")
    print(f"   • Accuracy:    {best_model['Accuracy']:.4f}")
    print(f"   • Precision:   {best_model['Precision']:.4f}")
    print(f"   • Recall:      {best_model['Recall']:.4f}")
    
    print(f"\n📊 RANKING DE MODELOS (por F1-Score):")
    for idx, row in df_comparison.iterrows():
        rank = idx + 1
        print(f"   {rank}. {row['Modelo']}: {row['F1-Score']:.4f}")
    
    print(f"\n📈 ESTADÍSTICAS GENERALES:")
    print(f"   • Mejor F1-Score:    {df_comparison['F1-Score'].max():.4f}")
    print(f"   • Promedio F1-Score: {df_comparison['F1-Score'].mean():.4f}")
    print(f"   • Mejor Accuracy:    {df_comparison['Accuracy'].max():.4f}")
    print(f"   • Promedio Accuracy: {df_comparison['Accuracy'].mean():.4f}")
    
    # Guardar reporte
    df_comparison.to_csv("comparacion_final_resultados.csv", index=False)
    print(f"\n✅ Resultados guardados en: comparacion_final_resultados.csv")


def main():
    """Función principal"""
    print("=" * 70)
    print("COMPARACIÓN FINAL DE TODOS LOS MODELOS")
    print("=" * 70)
    
    # Cargar resultados consolidados
    print("\n📂 Cargando resultados de los experimentos...")
    all_results = create_consolidated_results()
    
    if not all_results:
        print("\n❌ No se pudieron cargar los resultados.")
        print("   Asegúrate de haber ejecutado los experimentos primero.")
        return
    
    # Crear tabla comparativa
    print("\n📊 Creando tabla comparativa...")
    df_comparison = create_comparison_table(all_results)
    
    if df_comparison is not None:
        print("\n📋 Tabla Comparativa:")
        print(df_comparison.to_string(index=False))
        
        # Visualizar comparación
        print("\n📈 Generando visualizaciones...")
        visualize_comparison(df_comparison)
        
        # Generar reporte
        generate_summary_report(df_comparison)
        
        print("\n" + "=" * 70)
        print("✅ COMPARACIÓN FINAL COMPLETADA")
        print("=" * 70)
        print("\n📁 Archivos generados:")
        print("   • comparacion_final_modelos.png")
        print("   • comparacion_final_resultados.csv")
    else:
        print("\n❌ No se pudo crear la tabla comparativa")


if __name__ == "__main__":
    main()

