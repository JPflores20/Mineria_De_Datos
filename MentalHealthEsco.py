import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import missingno as msno
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib

df = pd.read_csv(r'C:\Users\VICTOR CORREA\Desktop\survey1.csv')
print(df.head())


#------ Columnas ------
#timestamp: fecha y hora de la entrevista
#age: edad del empleado
#gender: genero del empleado
#country: pais donde radica el empleado
#state: estado donde radica el empleado
#self_employee: si trabaja por cuenta propia
#family_history: antecedentes familiares
#treatment: ha buscado tratamiento para alguna enfermedad  <------- dependiente
#work_interference: su enfermedad mental interfiere su trabajo
#no_employees: cuantos empleados tiene la empresa
#remote_work: trabaa a distancia (al menos al 50%)
#tech_company: su empleador es una empresa tecnologica
#benefits: la empresa tiene prestaciones de salud mental?
#care_options: conoce las opciones de atencion a la salud mental que ofrece su empresa
#wellness_program: ha hablado alguna vez de su empresa de la salud mental como parte de un programa de bienestar para empleados
#seek_help: proporciona su empresa recursos para aprender sobre los problemas mentales
#anymity: esta protegido ti identidad si decides aprovechar los recursosde tratamiento de salud mental
#leave: le resulta facil solicitar una baja medica por problemas de salud mental
#mentalhealthconsequence: hablar de salud mental le traeria consecuencias negativas en tu empresa?
#phyhealthconsequence: hablar de salud fisica le traeria consecuencias negativas con tu jefe?
#coworkers: hablarias de un problema de salud mental con tus compañeros?
#phyhealthinterview: hablarias de un problemas de salud fisica con un posible empleador een una entrevista
#mentalvsphysical: crees que tu empresa se toma la salud mental igual de serio como la fisica?
#obs_consequence:  
#comments: comentario adicional
df.info()
""" Inferencia
- Hay un total de 27 columnas 
- Todas son de tipo de object excepto la edad
- La columna comments tiene muchos valores nulos, esto debido a que es un cuadro de texto opcional.
- Eliminaremos las columna timestamp por que contiene la fecha el mes y el año que fue tomado la encuesta, por lo tanto es irrelvante
- La columan de estado tiene muchos valores nulos
"""
print(df['Country'].value_counts())
print("\n")
print(df['state'].unique())
""" Inferencia
Seria engañoso concluir que un determinado pais tiene mas problemasde salud ental de los empleados porque alrededor del 60% de las personas pertenecen de Estados Unidos
Hay muchos paises con un solo encuestado, por lo tanto la columna pais carece de sentido.(Eliminar)
Los estados corresponden solamente a Estados Unidos.(Eliminar)
"""
df.drop(columns=['Timestamp','Country','state','comments'],inplace=True)
df.info()

# Preparacion de los datos
print("El conjunto de datos de datos contiene diferentes:\n")
print(df['Age'].unique())
print("Las notaciones de los generos son:\n ")
print(df['Gender'].unique())
""" Inferencia
Tenemos edades negativas, lo cual no tiene sentido
Tenemos edades fuera de la distribucion de los datos
Debemos establecer una edad minima laboral
"""

df.drop(df[df['Age'] < 16].index, inplace=True)
df.drop(df[df['Age'] > 100].index, inplace=True)
print(df['Age'].unique())

df['Gender'].replace(['Male ', 'male', 'M', 'm', 'Male', 'Cis Male','Man', 'cis male', 'Mail', 'Male-ish', 'Male (CIS)','Cis Man', 'msle', 'Malr', 'Mal', 'maile', 'Make',], 'Male', inplace = True)

df['Gender'].replace(['Female ', 'female', 'F', 'f', 'Woman', 'Female','femail', 'Cis Female', 'cis-female/femme', 'Femake', 'Female (cis)','woman',], 'Female', inplace = True)

df["Gender"].replace(['Female (trans)', 'queer/she/they', 'non-binary','fluid', 'queer', 'Androgyne', 'Trans-female', 'male leaning androgynous','Agender', 'A little about you', 'Nah', 'All','ostensibly male, unsure what that really means','Genderqueer', 'Enby', 'p', 'Neuter', 'something kinda male?','Guy (-ish) ^_^', 'Trans woman',], 'Other', inplace = True)

df['Gender'].value_counts()

"""Inferencias
El numero de hombres en el conjunto de datos es 4 vecees superior a de mujeres.
 Por lo tanto, debemos de tener en cuenta esto y evitar hacer suposiciones erroneas como que los hombres son mas suceptibes a a tener problemas de salud mental.
Tambien podemos cuncluir que el numero de hombres en la industria tecnologica es mucho mayor que el de las mujeres.
"""
#Grafica de datos para ver los valores faltantes


sns.set_style('dark')
color = ['grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey', 'grey','grey','grey','grey','grey','grey', 'grey','grey','grey','#D7BDE2','#D7BDE2']
msno.bar(df,fontsize =14, color = color, sort = 'descending', figsize = (15,10))

plt.text(0.05,1.265,'Mental Health at Workplace : Null Values', {'font':'serif', 'size':20, 'weight':'bold'})
plt.text(0.05,1.15,'''We have performed some feature engineering on our dataset. Now, let us try to see if there are any null values remaining in the dataset.''', {'font':'serif', 'size':12, 'weight':'normal'}, alpha = 0.8)
plt.xticks( rotation = 90,**{'font':'serif','size':14,'weight':'bold','horizontalalignment': 'center'},alpha = 0.8)

#plt.show()

# Grafica de las  personas que han buscado tratamiento para un problema de salud mental
sns.set_style("whitegrid")
plt.figure(figsize = (8,5))
plt.title('Get Treatment of Survey Respondents', fontsize=18, fontweight='bold')
eda_percentage = df['treatment'].value_counts(normalize = True).rename_axis('treatment').reset_index(name = 'Percentage')

ax = sns.barplot(x = 'treatment', y = 'Percentage', data = eda_percentage.head(10), palette='Purples')
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center', fontweight='bold')
#plt.show()

"""
Este es el resultado que obtienen los encuestados de la pregunta:
¿ha buscado tratamiento para un problema de salud mental?

Esta es nuestra variable objetivo. Si observamos el primer gráfico no hay mucha diferencia entre la cantidad de personas que si 
han buscado ayuda y las que no.
"""
"""

# Modo de trabajo y tratamiento

plt.figure(figsize = (20,6))
plt.subplot(1,2,1)
eda_percentage = df['self_employed'].value_counts(normalize = True).rename_axis('self_employed').reset_index(name = 'Percentage')
ax = sns.barplot(x = 'self_employed', y = 'Percentage', data = eda_percentage, palette = 'Purples')
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center', fontweight='bold')

plt.title('Employement Type of the Employees', fontsize=10, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.subplot(1,2,2)
sns.countplot(data = df, x=df['self_employed'],  hue = df['treatment'], palette = 'Purples')
plt.title('Employement Type of the Employees who are seeking Treatment',  fontsize=10, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.show()

# Antecedentes familiares
plt.figure(figsize = (20,6))
plt.subplot(1,2,1)
eda_percentage = df['family_history'].value_counts(normalize = True).rename_axis('family_history').reset_index(name = 'Percentage')
ax = sns.barplot(x = 'family_history', y = 'Percentage', data = eda_percentage, palette='Purples')
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center', fontweight='bold')

plt.title('Family History of Survey Respondents', fontsize=18, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.subplot(1,2,2)
sns.countplot(data = df, x=df['family_history'], hue = df['treatment'], palette='Purples')
plt.title('Family History of Survey Respondents', fontsize=18, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.show()

#Edad y tratamiento
plt.figure(figsize = (20,6))
plt.subplot(1,2,1)
eda_percentage = df['Age'].value_counts(normalize = True).rename_axis('Age').reset_index(name = 'Percentage')
ax = sns.barplot(x = 'Age', y = 'Percentage', data = eda_percentage, palette='Purples')
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center', fontweight='bold')

plt.title('Age of Survey Respondents', fontsize=18, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.subplot(1,2,2)
sns.countplot(data = df, x=df['Age'], hue = df['treatment'], palette='Purples')
plt.title('Age of Survey Respondents', fontsize=18, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16 )
plt.show()

#Genero y tratamiento
plt.figure(figsize = (20,6))
plt.subplot(1,2,1)
eda_percentage = df['Gender'].value_counts(normalize = True).rename_axis('Gender').reset_index(name = 'Percentage')
ax = sns.barplot(x = 'Gender', y = 'Percentage', data = eda_percentage, palette='Purples')
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center', fontweight='bold')

plt.title('Gender of Survey Respondents', fontsize=18, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.subplot(1,2,2)
sns.countplot(data = df, x=df['Gender'], hue = df['treatment'], palette='Purples')
plt.title('Gender of Survey Respondents', fontsize=18, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16 )
plt.show()

#Work interfere y tratamiento
plt.figure(figsize = (20,6))
plt.subplot(1,2,1)
eda_percentage = df['work_interfere'].value_counts(normalize = True).rename_axis('work_interfere').reset_index(name = 'Percentage')
ax = sns.barplot(x = 'work_interfere', y = 'Percentage', data = eda_percentage, palette='Purples')
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center', fontweight='bold')

plt.title('Work interfere of Survey Respondents', fontsize=18, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.subplot(1,2,2)
sns.countplot(data = df, x=df['work_interfere'], hue = df['treatment'], palette='Purples')
plt.title('Work interfere of Survey Respondents', fontsize=18, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16 )
plt.show()

# no employees y tratamiento
plt.figure(figsize = (20,6))
plt.subplot(1,2,1)
eda_percentage = df['no_employees'].value_counts(normalize = True).rename_axis('no_employees').reset_index(name = 'Percentage')
ax = sns.barplot(x = 'no_employees', y = 'Percentage', data = eda_percentage, palette='Purples')
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center', fontweight='bold')

plt.title('Number of employees in the companies of the survey respondents', fontsize=10, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.subplot(1,2,2)
sns.countplot(data = df, x=df['no_employees'], hue = df['treatment'], palette='Purples')
plt.title('Number of employees in the companies of the survey respondents', fontsize=10, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16 )
plt.show()

# remote work y tratamiento
plt.figure(figsize = (20,6))
plt.subplot(1,2,1)
eda_percentage = df['remote_work'].value_counts(normalize = True).rename_axis('remote_work').reset_index(name = 'Percentage')
ax = sns.barplot(x = 'remote_work', y = 'Percentage', data = eda_percentage, palette='Purples')
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center', fontweight='bold')

plt.title('Work type of Survey Respondents', fontsize=18, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.subplot(1,2,2)
sns.countplot(data = df, x=df['remote_work'], hue = df['treatment'], palette='Purples')
plt.title('Work type of Survey Respondents', fontsize=18, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16 )
plt.show()

# tech company y tratamiento
plt.figure(figsize = (20,6))
plt.subplot(1,2,1)
eda_percentage = df['tech_company'].value_counts(normalize = True).rename_axis('tech_company').reset_index(name = 'Percentage')
ax = sns.barplot(x = 'tech_company', y = 'Percentage', data = eda_percentage, palette='Purples')
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center', fontweight='bold')

plt.title('Is a Tech Company?', fontsize=18, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.subplot(1,2,2)
sns.countplot(data = df, x=df['tech_company'], hue = df['treatment'], palette='Purples')
plt.title('Is a Tech Company?', fontsize=18, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16 )
plt.show()

# Benefits y tratamiento
plt.figure(figsize = (20,6))
plt.subplot(1,2,1)
eda_percentage = df['benefits'].value_counts(normalize = True).rename_axis('benefits').reset_index(name = 'Percentage')
ax = sns.barplot(x = 'benefits', y = 'Percentage', data = eda_percentage, palette='Purples')
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center', fontweight='bold')

plt.title('Does the employee have benefits?', fontsize=14, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.subplot(1,2,2)
sns.countplot(data = df, x=df['benefits'], hue = df['treatment'], palette='Purples')
plt.title('Does the employee have benefits?', fontsize=14, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16 )
plt.show()

# Care options y tratamiento
plt.figure(figsize = (20,6))
plt.subplot(1,2,1)
eda_percentage = df['care_options'].value_counts(normalize = True).rename_axis('care_options').reset_index(name = 'Percentage')
ax = sns.barplot(x = 'care_options', y = 'Percentage', data = eda_percentage, palette='Purples')
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center', fontweight='bold')

plt.title('Does the employee have care options?', fontsize=14, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.subplot(1,2,2)
sns.countplot(data = df, x=df['care_options'], hue = df['treatment'], palette='Purples')
plt.title('Does the employee have care options?', fontsize=14, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16 )
plt.show()


# wellness program y tratamiento
plt.figure(figsize = (20,6))
plt.subplot(1,2,1)
eda_percentage = df['wellness_program'].value_counts(normalize = True).rename_axis('wellness_program').reset_index(name = 'Percentage')
ax = sns.barplot(x = 'wellness_program', y = 'Percentage', data = eda_percentage, palette='Purples')
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center', fontweight='bold')

plt.title('Wellness program', fontsize=14, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.subplot(1,2,2)
sns.countplot(data = df, x=df['wellness_program'], hue = df['treatment'], palette='Purples')
plt.title('Wellness program and treatment', fontsize=14, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16 )
plt.show()


# Seek help y tratamiento
plt.figure(figsize = (20,6))
plt.subplot(1,2,1)
eda_percentage = df['seek_help'].value_counts(normalize = True).rename_axis('seek_help').reset_index(name = 'Percentage')
ax = sns.barplot(x = 'seek_help', y = 'Percentage', data = eda_percentage, palette='Purples')
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center', fontweight='bold')

plt.title('Seek help', fontsize=14, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.subplot(1,2,2)
sns.countplot(data = df, x=df['seek_help'], hue = df['treatment'], palette='Purples')
plt.title('Seek help and treatment', fontsize=14, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16 )
plt.show()

# Anonimity y tratamiento
plt.figure(figsize = (20,6))
plt.subplot(1,2,1)
eda_percentage = df['anonymity'].value_counts(normalize = True).rename_axis('anonymity').reset_index(name = 'Percentage')
ax = sns.barplot(x = 'anonymity', y = 'Percentage', data = eda_percentage, palette='Purples')
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center', fontweight='bold')

plt.title('Anonymity', fontsize=14, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.subplot(1,2,2)
sns.countplot(data = df, x=df['anonymity'], hue = df['treatment'], palette='Purples')
plt.title('Anonymity and treatment', fontsize=14, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16 )
plt.show()

# leave y tratamiento
plt.figure(figsize = (20,6))
plt.subplot(1,2,1)
eda_percentage = df['leave'].value_counts(normalize = True).rename_axis('leave').reset_index(name = 'Percentage')
ax = sns.barplot(x = 'leave', y = 'Percentage', data = eda_percentage, palette='Purples')
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center', fontweight='bold')

plt.title('Leave', fontsize=14, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.subplot(1,2,2)
sns.countplot(data = df, x=df['leave'], hue = df['treatment'], palette='Purples')
plt.title('Leave and treatment', fontsize=14, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16 )
plt.show()

# Mental health consequence y tratamiento
plt.figure(figsize = (20,6))
plt.subplot(1,2,1)
eda_percentage = df['mental_health_consequence'].value_counts(normalize = True).rename_axis('mental_health_consequence').reset_index(name = 'Percentage')
ax = sns.barplot(x = 'mental_health_consequence', y = 'Percentage', data = eda_percentage, palette='Purples')
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center', fontweight='bold')

plt.title('Mental health consequence', fontsize=14, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.subplot(1,2,2)
sns.countplot(data = df, x=df['mental_health_consequence'], hue = df['treatment'], palette='Purples')
plt.title('Mental health consequence and treatment', fontsize=14, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16 )
plt.show()


# phys_health_consequence y tratamiento
plt.figure(figsize = (20,6))
plt.subplot(1,2,1)
eda_percentage = df['phys_health_consequence'].value_counts(normalize = True).rename_axis('phys_health_consequence').reset_index(name = 'Percentage')
ax = sns.barplot(x = 'phys_health_consequence', y = 'Percentage', data = eda_percentage, palette='Purples')
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center', fontweight='bold')

plt.title('Physical health consequences', fontsize=14, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.subplot(1,2,2)
sns.countplot(data = df, x=df['phys_health_consequence'], hue = df['treatment'], palette='Purples')
plt.title('Physical health consequences and treatment', fontsize=14, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16 )
plt.show()

# coworkers y tratamiento
plt.figure(figsize = (20,6))
plt.subplot(1,2,1)
eda_percentage = df['coworkers'].value_counts(normalize = True).rename_axis('coworkers').reset_index(name = 'Percentage')
ax = sns.barplot(x = 'coworkers', y = 'Percentage', data = eda_percentage, palette='Purples')
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center', fontweight='bold')

plt.title('Coworkers', fontsize=14, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.subplot(1,2,2)
sns.countplot(data = df, x=df['coworkers'], hue = df['treatment'], palette='Purples')
plt.title('Coworkers and treatment', fontsize=14, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16 )
plt.show()

# supervisoS y tratamiento
plt.figure(figsize = (20,6))
plt.subplot(1,2,1)
eda_percentage = df['supervisor'].value_counts(normalize = True).rename_axis('supervisor').reset_index(name = 'Percentage')
ax = sns.barplot(x = 'supervisor', y = 'Percentage', data = eda_percentage, palette='Purples')
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center', fontweight='bold')

plt.title('Supervisor', fontsize=14, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.subplot(1,2,2)
sns.countplot(data = df, x=df['supervisor'], hue = df['treatment'], palette='Purples')
plt.title('Supervisor and treatment', fontsize=14, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16 )
plt.show()

df.info()



# MHI y tratamiento
plt.figure(figsize = (20,6))
plt.subplot(1,2,1)
eda_percentage = df['mental_health_interview'].value_counts(normalize = True).rename_axis('mental_health_interview').reset_index(name = 'Percentage')
ax = sns.barplot(x = 'mental_health_interview', y = 'Percentage', data = eda_percentage, palette='Purples')
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center', fontweight='bold')

plt.title('Mental health interview', fontsize=14, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.subplot(1,2,2)
sns.countplot(data = df, x=df['mental_health_interview'], hue = df['treatment'], palette='Purples')
plt.title('Mental health interview and treatment', fontsize=14, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16 )
plt.show()

df.info()

# PHI y tratamiento
plt.figure(figsize = (20,6))
plt.subplot(1,2,1)
eda_percentage = df['phys_health_interview'].value_counts(normalize = True).rename_axis('phys_health_interview').reset_index(name = 'Percentage')
ax = sns.barplot(x = 'phys_health_interview', y = 'Percentage', data = eda_percentage, palette='Purples')
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center', fontweight='bold')

plt.title('Physical health interview', fontsize=14, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.subplot(1,2,2)
sns.countplot(data = df, x=df['phys_health_interview'], hue = df['treatment'], palette='Purples')
plt.title('Physical health interview and treatment', fontsize=14, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16 )
plt.show()

df.info()

# mVs y tratamiento
plt.figure(figsize = (20,6))
plt.subplot(1,2,1)
eda_percentage = df['mental_vs_physical'].value_counts(normalize = True).rename_axis('mental_vs_physical').reset_index(name = 'Percentage')
ax = sns.barplot(x = 'mental_vs_physical', y = 'Percentage', data = eda_percentage, palette='Purples')
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center', fontweight='bold')

plt.title('Mental vs Physical health interview', fontsize=10, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.subplot(1,2,2)
sns.countplot(data = df, x=df['mental_vs_physical'], hue = df['treatment'], palette='Purples')
plt.title('Mental vs Physical health interview and treatment', fontsize=10, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16 )
plt.show()

df.info()

# obs_consequence y tratamiento
plt.figure(figsize = (20,6))
plt.subplot(1,2,1)
eda_percentage = df['obs_consequence'].value_counts(normalize = True).rename_axis('obs_consequence').reset_index(name = 'Percentage')
ax = sns.barplot(x = 'obs_consequence', y = 'Percentage', data = eda_percentage, palette='Purples')
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center', fontweight='bold')

plt.title('Observed consequences interview', fontsize=10, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.subplot(1,2,2)
sns.countplot(data = df, x=df['obs_consequence'], hue = df['treatment'], palette='Purples')
plt.title('Observed consequences interview and treatment', fontsize=10, fontweight='bold')
plt.xticks(fontsize=7)
plt.yticks(fontsize=16)
plt.legend(fontsize=16 )
plt.show()

df.info()
"""

#Preparacion de los datos

df.info()
"""
tenemos solamente dos columnas que contienen valores faltantes 
work_interfere y self_employee
Vamos a intenter rellenar esos valoes y dejar nuestros datos listos para su procesamiento
- Hay solamnete 20% de NaN en la variables work_interference, esto lo cambieremos por "Dont know"

- Hay solamente un 1.4% de Nan en la variables en la variable self_employed, los vamos a cambiar por NOT

"""
df['work_interfere'] = df['work_interfere'].fillna('Don\'t know')
print(df['work_interfere'].unique())

df['self_employed'] = df['self_employed'].fillna('No')
print(df['self_employed'].unique())


print(df.isnull().sum())
print(df.columns)

list_col= ['Age', 'Gender', 'self_employed', 'family_history', 'treatment',
       'work_interfere', 'no_employees', 'remote_work', 'tech_company',
       'benefits', 'care_options', 'wellness_program', 'seek_help',
       'anonymity', 'leave', 'mental_health_consequence',
       'phys_health_consequence', 'coworkers', 'supervisor',
       'mental_health_interview', 'phys_health_interview',
       'mental_vs_physical', 'obs_consequence']


for col in list_col:
    print('{} :{}'.format(col.upper(),df[col].unique()))

#Label Encoder the categorical variables
from sklearn.preprocessing import LabelEncoder
object_cols= ['Gender', 'self_employed', 'family_history', 'treatment',
       'work_interfere', 'no_employees', 'remote_work', 'tech_company',
       'benefits', 'care_options', 'wellness_program', 'seek_help',
       'anonymity', 'leave', 'mental_health_consequence',
       'phys_health_consequence', 'coworkers', 'supervisor',
       'mental_health_interview', 'phys_health_interview',
       'mental_vs_physical', 'obs_consequence']

labelEncoder = LabelEncoder()
for col in object_cols:
    labelEncoder.fit(df[col])
    df[col] = labelEncoder.transform(df[col])

for col in list_col:
    print('{} :{}'.format(col.upper(),df[col].unique()))
    

df['treatment'].value_counts()

df['Gender'].value_counts()

print("sss")
print(df['tech_company'].value_counts())


#Genrear la matriz de correlacion
corr = df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
f,ax = plt.subplots(figsize= (30,30))
#Dibujar
sns.heatmap(corr, mask= mask, cmap = 'Oranges', vmax=0.3, center  =0, square=True, linewidths=0.5, cbar_kws={"shrink":0.5},annot=True)
plt.show()

def matrizDeConfusion(cf_matrix,name):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Oranges', cbar=False,
                xticklabels=['Predicción No', 'Predicción Sí'],
                yticklabels=['Actual No', 'Actual Sí'])
    plt.title(f'Matriz de Confusión - {name}')
    plt.xlabel('Predicciones')
    plt.ylabel('Valores Actuales')
    plt.show()
    plt.close()
    
import time
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_curve, confusion_matrix, classification_report, precision_recall_curve,auc,f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Datos y división de entrenamiento-prueba
X = df.drop('treatment', axis=1)
y = df['treatment']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=101)
print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('x_test shape: ', X_test.shape)
print('y_test shape: ', y_test.shape)


# Modelos
key = ['LogisticRegression', 'KNeighborsClassifier', 'DecisionTreeClassifier','RandomForestClassifier','Naive Bayes']
value = [LogisticRegression(), KNeighborsClassifier(n_neighbors=7),
        DecisionTreeClassifier(random_state=10),RandomForestClassifier(n_estimators=325, random_state=0), GaussianNB()]
models = dict(zip(key, value))
print(models)


predicted = []
# ------------- Tarea Clasificación mental health ----------------------
# Ajuste, predicción y cálculo de métricas
for name, algoritmo in models.items():
    model = algoritmo
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    tiempo = end_time - start_time
    predict = model.predict(X_test)
    cf_matrix = confusion_matrix(y_test, predict)

    # Cálculo de métricas

    acc = accuracy_score(y_test, predict)
    recall = recall_score(y_test, predict)
    specificity = cf_matrix[0, 0] / (cf_matrix[0, 0] + cf_matrix[0, 1])
    f1 = f1_score(y_test, predict)

    predicted.append(acc)

    #print(cf_matrix)
    fpr, tpr, thresholds = roc_curve(y_test, predict)
    roc_auc = auc(fpr, tpr)

    print(f'Nombre: {name} Acc: {acc} recall {recall}  especificidad {specificity}  f1 {f1} AUC {roc_auc}tiempo {tiempo}')


    plt.plot(fpr, tpr, label=f'{name} ROC curve (area = %0.2f)' %roc_auc)
   
#Graficar curvas ROC
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.rcParams['font.size'] = 12
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontweight='bold', fontsize=14)
plt.ylabel('Tasa de Verdaderos Positivos (Sensitividad)', fontweight='bold', fontsize=14)
plt.title('Curva ROC Comparativa', fontweight='bold', fontsize=16)
plt.legend(loc="lower right")
plt.show()

# Matrices de confusion
for name, algoritmo in models.items():
    predict = algoritmo.predict(X_test)
    cf_matrix = confusion_matrix(y_test, predict)
    matrizDeConfusion(cf_matrix, name)


#Mejor k
k_optimo=0
mejor_acc =0
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    
    acc = accuracy_score(y_test, predictions)
    if acc > mejor_acc:
        mejor_acc=acc
        k_optimo=k
    print(f'k = {k}, Accuracy = {acc}')
print("El mejor accuracy es ", mejor_acc, "cuando el valor de k es ",k_optimo)



#mejor n random forest
"""
n_optimo=0
mejor_acc =0
for n in range(100, 501):
    ranfor = RandomForestClassifier(n_estimators=n, random_state=0)
    ranfor.fit(X_train, y_train)
    predictions = ranfor.predict(X_test)
    
    acc = accuracy_score(y_test, predictions)
    if acc > mejor_acc:
        mejor_acc=acc
        n_optimo=n
    print(f'n = {n}, Accuracy = {acc}')
print("El mejor accuracy es ", mejor_acc, "cuando el valor de n es ",n_optimo)
"""

df_confirmada = df[['family_history', 'work_interfere']]

df_ultimas5 = df[['tech_company', 'self_employed', 'mental_health_interview', 'remote_work', 'phys_health_consequence']]

df_primeras5 = df[['Age', 'care_options', 'benefits', 'no_employees','leave']]

df_rechazados = df[['Age', 'Gender', 'self_employed', 'no_employees', 'remote_work', 'tech_company', 
                    'benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave', 
                    'mental_health_consequence', 'phys_health_consequence', 'coworkers', 'supervisor', 
                    'mental_health_interview', 'phys_health_interview', 'mental_vs_physical', 'obs_consequence']]


print("------- Confirmadas --------")
X_train, X_test, y_train, y_test = train_test_split(df_confirmada, y, stratify=y, test_size=0.3, random_state=101)
# ------------- Tarea Clasificación mental health ----------------------
# Ajuste, predicción y cálculo de métricas
for name, algoritmo in models.items():
    model = algoritmo
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    tiempo = end_time - start_time
    predict = model.predict(X_test)
    cf_matrix = confusion_matrix(y_test, predict)

    # Cálculo de métricas

    acc = accuracy_score(y_test, predict)
    recall = recall_score(y_test, predict)
    specificity = cf_matrix[0, 0] / (cf_matrix[0, 0] + cf_matrix[0, 1])
    f1 = f1_score(y_test, predict)

    predicted.append(acc)

    #print(cf_matrix)
    fpr, tpr, thresholds = roc_curve(y_test, predict)
    roc_auc = auc(fpr, tpr)

    print(f'Nombre: {name} Acc: {acc} recall {recall}  especificidad {specificity}  f1 {f1} AUC {roc_auc}tiempo {tiempo}')


    plt.plot(fpr, tpr, label=f'{name} ROC curve (area = %0.2f)' %roc_auc)
   
#Graficar curvas ROC
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.rcParams['font.size'] = 12
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontweight='bold', fontsize=14)
plt.ylabel('Tasa de Verdaderos Positivos (Sensitividad)', fontweight='bold', fontsize=14)
plt.title('Curva ROC Comparativa', fontweight='bold', fontsize=16)
plt.legend(loc="lower right")
plt.show()


#Selector de caracteristicas Boruta
from boruta import BorutaPy
model = RandomForestClassifier(random_state=42)
feat_selector = BorutaPy(model,n_estimators='auto', verbose=2, random_state=0)

feat_selector.fit(X,y)

feat_selector.ranking_

#Imprimir caracteristicas y ranking para cada vairable
print("\n ------- RANKING FOR EACH FEATURE --------")
for i in range(len(feat_selector.support_)):
    if feat_selector.support_[i]:
        print("Pasó la prueba", X.columns[i], "- Ranking: ", feat_selector.ranking_[i])
    else:
        print("No pasó la prueba: ", X.columns[i], "- Ranking: ", feat_selector.ranking_[i])

print("Columnas en df:", df.columns)


def evaluar_modelos(X, y, nombre_df):
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=101)
    
    print(f"Evaluación para el DataFrame: {nombre_df}")
    
    for name, algoritmo in models.items():
        model = algoritmo
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        tiempo = end_time - start_time
        predict = model.predict(X_test)
        cf_matrix = confusion_matrix(y_test, predict)

        # Cálculo de métricas
        acc = accuracy_score(y_test, predict)
        recall = recall_score(y_test, predict)
        specificity = cf_matrix[0, 0] / (cf_matrix[0, 0] + cf_matrix[0, 1])
        f1 = f1_score(y_test, predict)

        # Calcular la curva ROC
        fpr, tpr, thresholds = roc_curve(y_test, predict)
        roc_auc = auc(fpr, tpr)

        print(f'Nombre: {name} Acc: {acc:.5f} recall: {recall:.5f} especificidad: {specificity:.5f} f1: {f1:.5f} AUC: {roc_auc:.5f} tiempo: {tiempo:.5f}')

        plt.plot(fpr, tpr, label=f'{name} ROC curve (area = %0.2f)' % roc_auc)

    # Graficar todas las curvas ROC
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.rcParams['font.size'] = 12
    plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontweight='bold', fontsize=14)
    plt.ylabel('Tasa de Verdaderos Positivos (Sensitividad)', fontweight='bold', fontsize=14)
    plt.title(f'Curva ROC Comparativa - {nombre_df}', fontweight='bold', fontsize=16)
    plt.legend(loc="lower right")
    plt.show()

# Llamadas a la función para cada DataFrame
evaluar_modelos(df_confirmada, y, "Confirmados")
evaluar_modelos(df_ultimas5, y, "Ultimos 5")
evaluar_modelos(df_primeras5, y, "Primeras 5")
evaluar_modelos(df_rechazados, y, "Rechazados")

