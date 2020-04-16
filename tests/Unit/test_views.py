from flask import url_for
from models import User, Video
import constants
import shutil
import os

class TestPage(object):
    def test_home_page(self, client):
        response = client.get(url_for('home'))
        assert response.status_code == 200

    def test_user_creation_and_deletion(self, db):
        new_user = User("test_user@gmail.com")

        db.session.add(new_user)
        db.session.commit()

        assert db.session.query(User).one()
        
        user = User.query.filter_by(email="test_user@gmail.com").first()
        db.session.delete(user)
        db.session.commit()
        user = User.query.filter_by(email="test_user@gmail.com").first()

        assert user == None

    def test_video_creation_and_deletion(self, db):
        new_user = User("test_user@gmail.com")
        db.session.add(new_user)
        db.session.commit()

        user = User.query.filter_by(email="test_user@gmail.com").first()
        new_vid = Video(user.id, "Title")

        db.session.add(new_vid)
        db.session.commit()

        assert db.session.query(Video).one()
        id = user.id
        db.session.delete(user)
        db.session.commit()
        user = User.query.filter_by(email="test_user@gmail.com").first()

        assert user == None

        videos = Video.query.filter_by(user_id=id).all()

        assert len(videos) == 0

    def test_dashboard(self, client, db):
        new_user = User(email="test_user@gmail.com")
        db.session.add(new_user)
        db.session.commit()
        with client.session_transaction() as sess:
            sess[constants.JWT_PAYLOAD] = {
                "email" : new_user.email
            }
            sess[constants.PROFILE_KEY] = {
                "user_id" : "test_user",
                "name" : "Test User"
            }

        res = client.get(url_for('dashboard'))
        assert res.status_code == 200
        assert "Welcome" in str(res.data)
        user = User.query.filter_by(email="test_user@gmail.com").first()
        db.session.delete(user)
        db.session.commit()
