#!/bin/bash
#
#SBATCH --job-name=hawkes_analysis
#SBATCH --output=/raid/home/students/regna_enz/SignatureMMDTesting/out_hawkes.out
#SBATCH --error=/raid/home/students/regna_enz/SignatureMMDTesting/out_hawkes.out

## Mails
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%VOTRE_EMAIL%

#SBATCH --partition=prod40

## 3g.40gb:1 pour prod40
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1

## total requested cpus (ntasks * cpus-per-task) must be in [1: 4 * nb_3g.40gb]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:0:0

## Activer l'environnement virtuel
source /raid/home/students/regna_enz/SignatureMMDTesting/.venv/bin/activate

## Aller dans le répertoire de travail
cd /raid/home/students/regna_enz/SignatureMMDTesting

## Lancer le script
python run_alpha4_scaling.py
