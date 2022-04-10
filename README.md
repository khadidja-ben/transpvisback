Transpvisback PFE Project 🚀
cmd : 
1- docker-compose build 
2- docker-compose up 
3- docker container ls 
4- docker exec -it container-id python manage.py makemigrations authentification # use the container-id of the image transpvisback_api-transpvisback 
5- docker exec -it container-id python manage.py makemigrations transparency
6- docker exec -it container-id python manage.py migrate # Initialize database
7- docker exec -it container-id python manage.py createsuperuser # create the super user to administrate the site, Prompts for username and password