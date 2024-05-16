# Generated by Django 5.0.3 on 2024-04-17 05:40

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("automation", "0003_algorithmselection"),
    ]

    operations = [
        migrations.RenameField(
            model_name="algorithmselection",
            old_name="Linear_Regression",
            new_name="linear_Regression",
        ),
        migrations.RenameField(
            model_name="algorithmselection",
            old_name="Logistic_Regression",
            new_name="logistic_Regression",
        ),
        migrations.RenameField(
            model_name="algorithmselection",
            old_name="Naive_Bayes",
            new_name="naive_Bayes",
        ),
        migrations.RenameField(
            model_name="algorithmselection",
            old_name="Random_Forest",
            new_name="random_Forest",
        ),
        migrations.RenameField(
            model_name="algorithmselection",
            old_name="Support_Vector_Machines",
            new_name="support_Vector_Machines",
        ),
    ]