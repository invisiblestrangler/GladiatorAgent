from gladiator.todo import TodoManager


def test_todo_roundtrip(tmp_path):
    path = tmp_path / "todo.json"
    manager = TodoManager(path)
    first = manager.add("inspect failure")
    manager.add("run tests")
    assert manager.open_count == 2
    manager.mark_done(first.id)
    assert TodoManager(path).open_count == 1
    assert "inspect failure" in TodoManager(path).render()


def test_todo_clear(tmp_path):
    path = tmp_path / "todo.json"
    manager = TodoManager(path)
    manager.add("temporary")
    manager.clear()
    assert manager.list() == []
