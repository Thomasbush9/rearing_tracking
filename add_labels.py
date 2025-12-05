from behavex.project.project import Project
from pathlib import Path



if __name__ == "__main__":

    project_dir_path = Path("/Users/thomasbush/Downloads/project_dir_rear_test")

    project = Project(project_dir = project_dir_path)
    project.annotate_sessions()



