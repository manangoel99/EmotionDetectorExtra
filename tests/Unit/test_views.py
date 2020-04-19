from flask import url_for
from models import User, Video
import constants
import shutil
import os
import io

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

    def test_video_upload(self, client, db):
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
        
        data = {}
        data['file'] = (io.BytesIO(b"abcdef"), "./testvdo.mp4")

        response = client.post(
            url_for('upload_file'),
            data=data,
            follow_redirects=True,
            content_type='multipart/form-data'
        )

        vid = Video.query.all()[-1]

        user = User.query.filter_by(email="test_user@gmail.com").first()

        assert vid.user_id == user.id
        assert vid.video_path == os.path.join(str(user.id), str(vid.id))

        shutil.rmtree(os.path.join("/home/app/public", str(user.id)))
        db.session.delete(user)
        db.session.commit()


