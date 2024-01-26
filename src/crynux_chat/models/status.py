from enum import IntEnum


class TaskStatus(IntEnum):
	TaskPending = 0
	TaskTransactionSent = 1
	TaskBlockchainConfirmed = 2
	TaskParamsUploaded = 3
	TaskPendingResult = 4
	TaskAborted = 5
	TaskSuccess = 6
